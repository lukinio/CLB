import ipdb
import math
from enum import Enum
from typing import Optional, cast

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
    def __init__(self, in_features: int, out_features: int, normalize_shift: bool) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.bias = Parameter(torch.empty(out_features))
        self._radius = Parameter(torch.empty((out_features, in_features + 1)), requires_grad=False)
        self.register_buffer("radius", torch.ones((out_features, in_features + 1)))
        self.radius_multiplier: Optional[Tensor] = None

        self._shift = Parameter(torch.empty((out_features, in_features + 1)), requires_grad=False)
        self._scale = Parameter(torch.empty((out_features, in_features + 1)), requires_grad=False)
        self.normalize_shift = normalize_shift

        self.mode: Mode = Mode.VANILLA
        self.previous_intervals = []

        self.reset_parameters()

    def radius_transform(self, params: Tensor):
        assert self.radius_multiplier is not None
        return torch.sigmoid(params) * self.radius_multiplier

    # @property
    # def radius(self) -> Tensor:
    #     return self.radius_transform(self._radius)

    @property
    def shift(self) -> Tensor:
        """ Contracted interval middle shift (-1, 1). """
        if self.normalize_shift:
            return (self._shift / (self.radius + 1e-8)).tanh()
        else:
            return self._shift.tanh()

    @property
    def scale(self) -> Tensor:
        """ Contracted interval scale (0, 1). """
        if True:
            return torch.zeros_like(self._scale)
        else:
            return self._scale.sigmoid()

    def clamp_radii(self):
        with torch.no_grad():
            assert self.radius_multiplier is not None
            max = 1 / float(self.radius_multiplier)
            self._radius.clamp_(min=0, max=max)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        with torch.no_grad():
            # -inf?
            self._radius.fill_(-10.)
            # self._radius.zero_()
            self._shift.zero_()
            self._scale.fill_(5)

            self.bias.zero_()

    def switch_mode(self, mode: Mode) -> None:
        self.mode = mode

        if mode == Mode.VANILLA:
            self.weight.requires_grad = True
            self.bias.requires_grad = True
            self._radius.requires_grad = False
            self._shift.requires_grad = False
            self._scale.requires_grad = False
        elif mode == Mode.EXPANSION:
            self.weight.requires_grad = False
            self.bias.requires_grad = False
            self._radius.requires_grad = True
            self._shift.requires_grad = False
            self._scale.requires_grad = False
        elif mode == Mode.CONTRACTION:
            self.weight.requires_grad = False
            self.bias.requires_grad = False
            self._radius.requires_grad = False
            self._shift.requires_grad = True
            self._scale.requires_grad = True

    def freeze_task(self) -> None:
        with torch.no_grad():

            self.prev_weight = self.weight.clone().detach()
            self.prev_bias = self.bias.clone().detach()
            self.weight.copy_(
                    self.weight
                    + self.shift[:, :-1]
                    * (torch.tensor(1.0) - self.scale[:, :-1]) * self.radius[:, :-1])
            self.bias.copy_(
                    self.bias
                    + self.shift[:, -1]
                    * (torch.tensor(1.0) - self.scale[:, -1]) * self.radius[:, -1])
            for (weight_interval, bias_interval) in self.previous_intervals:
                try:
                    assert (weight_interval[0] - 1e-5 <= self.weight).all()
                    assert (weight_interval[1] + 1e-5 >= self.weight).all()
                    assert (bias_interval[0] - 1e-5 <= self.bias).all()
                    assert (bias_interval[1] + 1e-5 >= self.bias).all()
                except AssertionError:
                    ipdb.set_trace()

            self._radius.copy_(self.scale * self._radius)
            self._shift.zero_()
            self._scale.fill_(5)

    def save_intervals(self):
        lower_weight_interval = self.weight - self.radius[:, :-1]
        upper_weight_interval = self.weight + self.radius[:, :-1]
        weight_interval = torch.stack(
                [lower_weight_interval, upper_weight_interval],
                dim=0)
        lower_bias_interval = self.bias - self.radius[:, -1]
        upper_bias_interval = self.bias + self.radius[:, -1]
        bias_interval = torch.stack(
                [lower_bias_interval, upper_bias_interval],
                dim=0)
        self.previous_intervals += [(weight_interval, bias_interval)]
        for (prev_weight_interval, prev_bias_interval) in self.previous_intervals[:-1]:
            try:
                assert (prev_weight_interval[0] - 1e-5 <= lower_weight_interval).all()
                assert (prev_weight_interval[1] + 1e-5 >= upper_weight_interval).all()
                assert (prev_bias_interval[0] - 1e-5 <= lower_bias_interval).all()
                assert (prev_bias_interval[1] + 1e-5 >= upper_bias_interval).all()
            except AssertionError:
                ipdb.set_trace()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = x.refine_names('N', 'bounds', 'features')  # type: ignore
        assert (x.rename(None) >= 0.0).all(), 'All input features must be non-negative.'  # type: ignore

        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind('bounds'))  # type: ignore
        assert (x_lower <= x_middle).all(), 'Lower bound must be less than or equal to middle bound.'
        assert (x_middle <= x_upper).all(), 'Middle bound must be less than or equal to upper bound.'

        if self.mode in [Mode.VANILLA, Mode.EXPANSION]:
            w_middle = self.weight
            w_lower = self.weight - self.radius[:, :-1]
            w_upper = self.weight + self.radius[:, :-1]

            b_middle = self.bias
            b_lower = self.bias - self.radius[:, -1]
            b_upper = self.bias + self.radius[:, -1]
        else:
            assert self.mode == Mode.CONTRACTION
            assert (0.0 <= self.scale).all() and (self.scale <= 1.0).all(), 'Scale must be in [0, 1] range.'
            if not ((-1.0 <= self.shift).all() and (self.shift <= 1.0).all()):
                ipdb.set_trace()
            assert (-1.0 <= self.shift).all() and (self.shift <= 1.0).all(), 'Shift must be in [-1, 1] range.'

            w_middle = (self.weight
                        + self.shift[:, :-1]
                        * (torch.tensor(1.0) - self.scale[:, :-1]) * self.radius[:, :-1])
            if True:
                w_lower = w_middle - self.radius[:, :-1]
                w_upper = w_middle + self.radius[:, :-1]
            else:
                w_lower = w_middle - self.scale[:, :-1] * self.radius[:, :-1]
                w_upper = w_middle + self.scale[:, :-1] * self.radius[:, :-1]

            b_middle = (self.bias
                        + self.shift[:, -1]
                        * (torch.tensor(1.0) - self.scale[:, -1]) * self.radius[:, -1])
            if True:
                b_lower = b_middle - self.radius[:, -1]
                b_upper = b_middle + self.radius[:, -1]
            else:
                b_lower = b_middle - self.scale[:, -1] * self.radius[:, :-1]
                b_upper = b_middle + self.scale[:, -1] * self.radius[:, -1]

        w_lower_pos = w_lower.clamp(min=0)
        w_lower_neg = w_lower.clamp(max=0)
        w_upper_pos = w_upper.clamp(min=0)
        w_upper_neg = w_upper.clamp(max=0)
        w_middle_pos = w_middle.clamp(min=0)  # split only needed for numeric stability with asserts
        w_middle_neg = w_middle.clamp(max=0)  # split only needed for numeric stability with asserts

        lower = x_lower @ w_lower_pos.t() + x_upper @ w_lower_neg.t()
        upper = x_upper @ w_upper_pos.t() + x_lower @ w_upper_neg.t()
        middle = x_middle @ w_middle_pos.t() + x_middle @ w_middle_neg.t()

        lower = lower + b_lower
        upper = upper + b_upper
        middle = middle + b_middle

        assert (lower <= middle).all(), 'Lower bound must be less than or equal to middle bound.'
        assert (middle <= upper).all(), 'Middle bound must be less than or equal to upper bound.'

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

    def freeze_task(self) -> None:
        for m in self.interval_children():
            m.freeze_task()

    def save_intervals(self) -> None:
        for m in self.interval_children():
            m.save_intervals()

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
    def __init__(self, input_size: int, hidden_dim: int, output_classes: int, normalize_shift: bool):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes
        self.normalize_shift = normalize_shift

        self.fc1 = IntervalLinear(self.input_size, self.hidden_dim, normalize_shift=normalize_shift)
        self.fc2 = IntervalLinear(self.hidden_dim, self.hidden_dim, normalize_shift=normalize_shift)
        self.last = IntervalLinear(self.hidden_dim, self.output_classes, normalize_shift=normalize_shift)

    def forward(self, x: Tensor) -> dict[str, Tensor]:  # type: ignore
        if len(x.shape) == 4:
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
