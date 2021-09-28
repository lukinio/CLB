import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torch import Tensor
from torch.nn.parameter import Parameter


class Mode(Enum):
    VANILLA = 0
    EXPANSION = 1
    CONTRACTION = 2


class IntervalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self._radius = Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self.radius_multiplier: Optional[Tensor] = None

        self._shift = Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self._scale = Parameter(torch.empty((out_features, in_features)), requires_grad=False)

        self.mode: Mode = Mode.VANILLA

        self.reset_parameters()

    def radius_transform(self, params: Tensor):
        assert self.radius_multiplier is not None
        return (params * self.radius_multiplier).clamp(min=0, max=1)

    @property
    def radius(self) -> Tensor:
        return self.radius_transform(self._radius)

    @property
    def shift(self) -> Tensor:
        """ Contracted interval middle shift (-1, 1). """
        return self._shift.tanh()

    @property
    def scale(self) -> Tensor:
        """ Contracted interval scale (0, 1). """
        return self._scale.sigmoid()

    def clamp_radii(self):
        with torch.no_grad():
            assert self.radius_multiplier is not None
            max = 1 / float(self.radius_multiplier)
            self._radius.clamp_(min=0, max=max)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        with torch.no_grad():
            self._radius.zero_()
            self._shift.zero_()
            self._scale.fill_(5)

    def switch_mode(self, mode: Mode) -> None:
        self.mode = mode

        if mode == Mode.VANILLA:
            self.weight.requires_grad = True
            self._radius.requires_grad = False
            self._shift.requires_grad = False
            self._scale.requires_grad = False
        elif mode == Mode.EXPANSION:
            self.weight.requires_grad = False
            self._radius.requires_grad = True
            self._shift.requires_grad = False
            self._scale.requires_grad = False
        elif mode == Mode.CONTRACTION:
            self.weight.requires_grad = False
            self._radius.requires_grad = False
            self._shift.requires_grad = True
            self._scale.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = x.refine_names('N', 'bounds', 'features')  # type: ignore
        assert (x >= 0.0).all(), 'All input features must be non-negative.'

        x_lower, x_middle, x_upper = x.unbind('bounds')
        assert (x_lower <= x_middle).all(), 'Lower bound must be less than or equal to middle bound.'
        assert (x_middle <= x_upper).all(), 'Middle bound must be less than or equal to upper bound.'

        if self.mode in [Mode.VANILLA, Mode.EXPANSION]:
            w_middle = self.weight
            w_lower = self.weight - self.radius
            w_upper = self.weight + self.radius
        else:
            assert self.mode == Mode.CONTRACTION
            assert all(0.0 <= self.scale <= 1.0), 'Scale must be in [0, 1] range.'
            assert all(-1.0 <= self.shift <= 1.0), 'Shift must be in [-1, 1] range.'

            w_middle = self.weight + self.shift * (torch.tensor(1.0) - self.scale) * self.radius
            w_lower = w_middle - self.scale * self.radius
            w_upper = w_middle + self.scale * self.radius

        w_lower_pos = w_lower.clamp(min=0)
        w_lower_neg = w_lower.clamp(max=0)
        w_upper_pos = w_upper.clamp(min=0)
        w_upper_neg = w_upper.clamp(max=0)
        w_middle_pos = w_middle.clamp(min=0)  # split only needed for numeric stability with asserts
        w_middle_neg = w_middle.clamp(max=0)  # split only needed for numeric stability with asserts

        lower = (x_lower @ w_lower_pos.t() + x_upper @ w_lower_neg.t()).rename(None)  # type: ignore
        upper = (x_upper @ w_upper_pos.t() + x_lower @ w_upper_neg.t()).rename(None)  # type: ignore
        middle = (x_middle @ w_middle_pos.t() + x_middle @ w_middle_neg.t()).rename(None)  # type: ignore

        assert (lower <= middle).all(), 'Lower bound must be less than or equal to middle bound.'  # type: ignore
        assert (middle <= upper).all(), 'Middle bound must be less than or equal to upper bound.'  # type: ignore

        return torch.stack([lower, middle, upper], dim=1).refine_names('N', 'bounds', 'features')  # type: ignore


class IntervalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.mode: Mode = Mode.VANILLA

    def interval_children(self) -> list[IntervalLinear]:
        return [m for m in self.children() if isinstance(m, IntervalLinear)]

    def named_interval_children(self) -> list[tuple[str, IntervalLinear]]:
        return [(n, m) for n, m in self.named_children() if isinstance(m, IntervalLinear)]

    def switch_mode(self, mode: Mode) -> None:
        if mode == Mode.VANILLA:
            print(f'\n[bold cyan]» :green_circle: Switching to vanilla training phase...')
        elif mode == Mode.EXPANSION:
            print(f'\n[bold cyan]» :yellow circle: Switching to interval expansion phase...')
        elif mode == Mode.CONTRACTION:
            print(f'\n[bold cyan]» :heavy_large_circle: Switching to interval contraction phase...')

        self.mode = mode
        for m in self.interval_children():
            m.switch_mode(mode)

    def set_radius_multiplier(self, multiplier: Tensor) -> None:
        for m in self.interval_children():
            m.radius_multiplier = multiplier

    def clamp_radii(self) -> None:
        for m in self.interval_children():
            m.clamp_radii()

    def radius_transform(self, params: Tensor) -> Tensor:
        for m in self.interval_children():
            return m.radius_transform(params)

        raise ValueError('No IntervalNet modules found in model.')


class IntervalMLP(IntervalModel):
    def __init__(self, input_size: int, hidden_dim: int, output_classes: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes

        self.fc1 = IntervalLinear(self.input_size, self.hidden_dim)
        self.fc2 = IntervalLinear(self.hidden_dim, self.hidden_dim)
        self.last = IntervalLinear(self.hidden_dim, self.output_classes)

    def forward(self, x: Tensor) -> dict[str, Tensor]:  # type: ignore
        x = x.refine_names('N', 'C', 'H', 'W')  # type: ignore  # expected input shape

        x = x.rename(None)  # type: ignore  # drop names for unsupported operations
        x = x.flatten(1)  # (N, features)
        x = x.unflatten(1, (1, -1))  # type: ignore  # (N, bounds, features)
        x = x.tile((1, 3, 1))

        x = x.refine_names('N', 'bounds', 'features')  # type: ignore

        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))
        last = self.last(fc2)

        return {
            'fc1': fc1,
            'fc2': fc2,
            'last': last,
        }

    @property
    def device(self):
        return self.fc1.weight.device
