import math
from abc import ABC
from enum import Enum
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.models import MultiTaskModule
from rich import print
from torch import Tensor
from torch.nn.parameter import Parameter


class Mode(Enum):
    VANILLA = 0
    EXPANSION = 1
    CONTRACTION_SHIFT = 2
    CONTRACTION_SCALE = 3


class IntervalModuleWithWeights(nn.Module, ABC):
    def __init__(self):
        super().__init__()


class PointLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.bias = Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        with torch.no_grad():
            self.bias.zero_()

    def forward(self, x):
        x = x.refine_names("N", "bounds", "features")  # type: ignore
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore

        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."

        w_middle_pos = self.weight.clamp(min=0)
        w_middle_neg = self.weight.clamp(max=0)

        lower = x_lower @ w_middle_pos.t() + x_upper @ w_middle_neg.t() + self.bias
        upper = x_upper @ w_middle_pos.t() + x_lower @ w_middle_neg.t() + self.bias
        middle = x_middle @ w_middle_pos.t() + x_middle @ w_middle_neg.t() + self.bias

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "features")  # type: ignore

    def switch_mode(self, mode: Mode) -> None:
        self.mode = mode

        def enable(params: list[Parameter]):
            for p in params:
                p.requires_grad_()

        def disable(params: list[Parameter]):
            for p in params:
                p.requires_grad_(False)
                p.grad = None

        disable([self.weight, self.bias])

        if mode == Mode.VANILLA:
            enable([self.weight, self.bias])
        elif mode == Mode.EXPANSION:
            pass
        elif mode == Mode.CONTRACTION_SHIFT:
            enable([self.weight, self.bias])
        elif mode == Mode.CONTRACTION_SCALE:
            pass


class IntervalLinear(IntervalModuleWithWeights):
    def __init__(
            self, in_features: int, out_features: int, radius_multiplier: float, max_radius: float, bias: bool,
            normalize_shift: bool, normalize_scale: bool, scale_init: float = -5.
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.radius_multiplier = radius_multiplier
        self.max_radius = max_radius
        self.bias = bias
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.scale_init = scale_init

        assert self.radius_multiplier > 0
        assert self.max_radius > 0

        if self.bias:
            in_features += 1

        self.weight = Parameter(torch.empty((out_features, in_features)))
        self._radius = Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self._shift = Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self._scale = Parameter(torch.empty((out_features, in_features)), requires_grad=False)

        self.mode: Mode = Mode.VANILLA

        self.reset_parameters()

    def radius_transform(self, params: Tensor):
        return (params * torch.tensor(self.radius_multiplier)).clamp(min=0, max=self.max_radius + 0.1)

    @property
    def radius(self) -> Tensor:
        return self.radius_transform(self._radius)

    @property
    def shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._shift.device)
            return (self._shift / torch.max(self.radius, eps)).tanh()
        else:
            return self._shift.tanh()

    @property
    def scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._shift.device)
            scale = (self._scale / torch.max(self.radius, eps)).sigmoid()
        else:
            scale = self._scale.sigmoid()

        return scale * (1.0 - torch.abs(self.shift))

    def clamp_radii(self) -> None:
        with torch.no_grad():
            max = self.max_radius / self.radius_multiplier
            self._radius.clamp_(min=0, max=max)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        # TODO: Adjust bias init
        with torch.no_grad():
            self._radius.fill_(self.max_radius)
            self._shift.zero_()
            self._scale.fill_(self.scale_init)

    def switch_mode(self, mode: Mode) -> None:
        self.mode = mode

        def enable(params: list[Parameter]):
            for p in params:
                p.requires_grad_()

        def disable(params: list[Parameter]):
            for p in params:
                p.requires_grad_(False)
                p.grad = None

        disable([self.weight, self._radius, self._shift, self._scale])

        if mode == Mode.VANILLA:
            enable([self.weight])
        elif mode == Mode.EXPANSION:
            with torch.no_grad():
                self._radius.fill_(self.max_radius)
            enable([self._radius])
        elif mode == Mode.CONTRACTION_SHIFT:
            enable([self._shift])
        elif mode == Mode.CONTRACTION_SCALE:
            enable([self._scale])

    def freeze_task(self) -> None:
        with torch.no_grad():
            self.weight.copy_(self.weight + self.shift * self.radius)
            self._radius.copy_(self.scale * self._radius)
            self._shift.zero_()
            self._scale.fill_(self.scale_init)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = x.refine_names("N", "bounds", "features")  # type: ignore
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore

        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."

        if self.mode in [Mode.VANILLA, Mode.EXPANSION]:
            w_middle: Tensor = self.weight
            w_lower = self.weight - self.radius
            w_upper = self.weight + self.radius
        else:
            assert self.mode in [Mode.CONTRACTION_SHIFT, Mode.CONTRACTION_SCALE]
            assert (0.0 <= self.scale).all() and (self.scale <= 1.0).all(), "Scale must be in [0, 1] range."
            assert (-1.0 <= self.shift).all() and (self.shift <= 1.0).all(), "Shift must be in [-1, 1] range."

            w_middle = self.weight + self.shift * self.radius
            w_lower = w_middle - self.scale * self.radius
            w_upper = w_middle + self.scale * self.radius

        w_lower_pos = (w_lower[:, :-1] if self.bias else w_lower).clamp(min=0)
        w_lower_neg = (w_lower[:, :-1] if self.bias else w_lower).clamp(max=0)
        w_upper_pos = (w_upper[:, :-1] if self.bias else w_upper).clamp(min=0)
        w_upper_neg = (w_upper[:, :-1] if self.bias else w_upper).clamp(max=0)
        # Further splits only needed for numeric stability with asserts
        w_middle_pos = (w_middle[:, :-1] if self.bias else w_middle).clamp(min=0)
        w_middle_neg = (w_middle[:, :-1] if self.bias else w_middle).clamp(max=0)

        lower = x_lower @ w_lower_pos.t() + x_upper @ w_lower_neg.t()
        upper = x_upper @ w_upper_pos.t() + x_lower @ w_upper_neg.t()
        middle = x_middle @ w_middle_pos.t() + x_middle @ w_middle_neg.t()

        if self.bias:
            lower = lower + w_lower[:, -1]
            upper = upper + w_upper[:, -1]
            middle = middle + w_middle[:, -1]

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "features")  # type: ignore


class IntervalDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.scale = 1. / (1 - self.p)

    def forward(self, x):
        if self.training:
            x = x.refine_names("N", "bounds", ...)
            x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)),
                                             x.unbind("bounds"))  # type: ignore
            mask = torch.bernoulli(self.p * torch.ones_like(x_middle)).long()
            x_lower = x_lower.where(mask != 1, torch.zeros_like(x_lower))
            x_middle = x_middle.where(mask != 1, torch.zeros_like(x_middle))
            x_upper = x_upper.where(mask != 1, torch.zeros_like(x_upper))
            return torch.stack([x_lower, x_middle, x_upper], dim=1)
        else:
            return x


class IntervalMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        x_lower = super().forward(x_lower)
        x_middle = super().forward(x_middle)
        x_upper = super().forward(x_upper)
        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                             "W")  # type: ignore


class IntervalAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super().__init__(output_size)

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        x_lower = super().forward(x_lower)
        x_middle = super().forward(x_middle)
        x_upper = super().forward(x_upper)
        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                             "W")  # type: ignore


class IntervalConv2d(nn.Conv2d, IntervalModuleWithWeights):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            radius_multiplier: float,
            max_radius: float,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
    ) -> None:
        IntervalModuleWithWeights.__init__(self)
        nn.Conv2d.__init__(self,in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.radius_multiplier = radius_multiplier
        self.max_radius = max_radius

        assert self.radius_multiplier > 0
        assert self.max_radius > 0

        self._radius = Parameter(torch.empty_like(self.weight), requires_grad=False)
        self._shift = Parameter(torch.empty_like(self.weight), requires_grad=False)
        self._scale = Parameter(torch.empty_like(self.weight), requires_grad=False)

        self._bias_radius = Parameter(torch.empty_like(self.bias), requires_grad=False)
        self._bias_shift = Parameter(torch.empty_like(self.bias), requires_grad=False)
        self._bias_scale = Parameter(torch.empty_like(self.bias), requires_grad=False)

        self.mode: Mode = Mode.VANILLA

        self.init_parameters()

    # TODO abstract away this logic in a common base class?
    def radius_transform(self, params: Tensor):
        return (params * torch.tensor(self.radius_multiplier)).clamp(min=0, max=self.max_radius)

    @property
    def radius(self) -> Tensor:
        return self.radius_transform(self._radius)

    @property
    def bias_radius(self) -> Tensor:
        return self.radius_transform(self._bias_radius)

    @property
    def shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        return self._shift.tanh()

    @property
    def bias_shift(self) -> Tensor:
        return self._bias_shift.tanh()

    @property
    def scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        return self._scale.sigmoid()

    @property
    def bias_scale(self) -> Tensor:
        return self._bias_scale.sigmoid()

    def clamp_radii(self) -> None:
        with torch.no_grad():
            max = self.max_radius / self.radius_multiplier
            self._radius.clamp_(min=0, max=max)
            self._bias_radius.clamp_(min=0, max=max)

    def init_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        # TODO: Adjust bias init
        with torch.no_grad():
            self._radius.zero_()
            self._shift.zero_()
            self._scale.fill_(5)
            self._bias_radius.zero_()
            self._bias_shift.zero_()
            self._bias_scale.fill_(5)

    def switch_mode(self, mode: Mode) -> None:
        self.mode = mode

        def enable(params: list[Parameter]):
            for p in params:
                p.requires_grad_()

        def disable(params: list[Parameter]):
            for p in params:
                p.requires_grad_(False)
                p.grad = None

        disable([self.weight, self._radius, self._shift, self._scale, self._bias_radius, self._bias_shift,
                 self._bias_scale])

        if mode == Mode.VANILLA:
            enable([self.weight])
        elif mode == Mode.EXPANSION:
            enable([self._radius, self._bias_radius])
        elif mode == Mode.CONTRACTION:
            enable([self._shift, self._scale, self._bias_shift, self._bias_scale])

    def freeze_task(self) -> None:
        with torch.no_grad():
            self.weight.copy_(self.weight + self.shift * (torch.tensor(1.0) - self.scale) * self.radius)
            self._radius.copy_(self.scale * self._radius)
            self._bias_radius.copy_(self.bias_scale * self._bias_radius)
            self._shift.zero_()
            self._bias_shift.zero_()
            self._scale.fill_(5)
            self._bias_scale.fill_(5)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = x.refine_names("N", "bounds", "C", "H", "W")
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."

        if self.mode in [Mode.VANILLA, Mode.EXPANSION]:
            w_middle: Tensor = self.weight
            w_lower = self.weight - self.radius
            w_upper = self.weight + self.radius
        else:
            assert self.mode == Mode.CONTRACTION
            assert (0.0 <= self.scale).all() and (self.scale <= 1.0).all(), "Scale must be in [0, 1] range."
            assert (-1.0 <= self.shift).all() and (self.shift <= 1.0).all(), "Shift must be in [-1, 1] range."

            w_middle = self.weight + self.shift * (torch.tensor(1.0) - self.scale) * self.radius
            w_lower = w_middle - self.scale * self.radius
            w_upper = w_middle + self.scale * self.radius

        w_lower_pos = w_lower.clamp(min=0)
        w_lower_neg = w_lower.clamp(max=0)
        w_upper_pos = w_upper.clamp(min=0)
        w_upper_neg = w_upper.clamp(max=0)
        # Further splits only needed for numeric stability with asserts
        w_middle_neg = w_middle.clamp(max=0)
        w_middle_pos = w_middle.clamp(min=0)

        l_lp = F.conv2d(x_lower, w_lower_pos, None, self.stride, self.padding, self.dilation, self.groups)
        u_ln = F.conv2d(x_upper, w_lower_neg, None, self.stride, self.padding, self.dilation, self.groups)
        u_up = F.conv2d(x_upper, w_upper_pos, None, self.stride, self.padding, self.dilation, self.groups)
        l_un = F.conv2d(x_lower, w_upper_neg, None, self.stride, self.padding, self.dilation, self.groups)
        m_mp = F.conv2d(x_middle, w_middle_pos, None, self.stride, self.padding, self.dilation, self.groups)
        m_mn = F.conv2d(x_middle, w_middle_neg, None, self.stride, self.padding, self.dilation, self.groups)

        lower = l_lp + u_ln
        upper = u_up + l_un
        middle = m_mp + m_mn

        if isinstance(self.bias, Parameter):
            b_middle = self.bias + self.bias_shift * (torch.tensor(1.0) - self.bias_scale) * self.bias_radius
            b_lower = b_middle - self.bias_scale * self.bias_radius
            b_upper = b_middle + self.bias_scale * self.bias_radius
            lower = lower + b_lower.view(1, b_lower.size(0), 1, 1)
            upper = upper + b_upper.view(1, b_upper.size(0), 1, 1)
            middle = middle + b_middle.view(1, b_middle.size(0), 1, 1)

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore


class IntervalModel(MultiTaskModule):
    def __init__(self, radius_multiplier: float, max_radius: float):
        super().__init__()

        self.mode: Mode = Mode.VANILLA
        self._radius_multiplier = radius_multiplier
        self._max_radius = max_radius

    def interval_children(self) -> list[IntervalModuleWithWeights]:
        return [m for m in self.modules() if isinstance(m, IntervalModuleWithWeights)]

    def named_interval_children(self) -> list[tuple[str, IntervalModuleWithWeights]]:
        # TODO: hack
        return [("last" if "last" in n else n, m)
                for n, m in self.named_modules()
                if isinstance(m, IntervalModuleWithWeights)]

    def switch_mode(self, mode: Mode) -> None:
        if mode == Mode.VANILLA:
            print("\n[bold cyan]» :green_circle: Switching to vanilla training phase...")
        elif mode == Mode.EXPANSION:
            print("\n[bold cyan]» :yellow circle: Switching to interval expansion phase...")
        elif mode == Mode.CONTRACTION_SHIFT:
            print("\n[bold cyan]» :heavy_large_circle: Switching to shift contraction phase...")
        elif mode == Mode.CONTRACTION_SCALE:
            print("\n[bold cyan]» :heavy_large_circle: Switching to scale contraction phase...")

        self.mode = mode
        for m in self.interval_children():
            m.switch_mode(mode)

        for m in self.last:
            if isinstance(m, PointLinear):
                m.switch_mode(mode)

    def freeze_task(self) -> None:
        for m in self.interval_children():
            m.freeze_task()

    @property
    def radius_multiplier(self):
        return self._radius_multiplier

    @radius_multiplier.setter
    def radius_multiplier(self, value: float):
        self._radius_multiplier = value
        for m in self.interval_children():
            m.radius_multiplier = value

    @property
    def max_radius(self):
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value: float) -> None:
        self._max_radius = value
        for m in self.interval_children():
            m.max_radius = value

    def clamp_radii(self) -> None:
        for m in self.interval_children():
            m.clamp_radii()

    def radius_transform(self, params: Tensor) -> Tensor:
        for m in self.interval_children():
            return m.radius_transform(params)

        raise ValueError("No IntervalNet modules found in model.")


class IntervalMLP(IntervalModel):
    def __init__(
            self,
            input_size: int,
            hidden_dim: int,
            output_classes: int,
            radius_multiplier: float,
            max_radius: float,
            bias: bool,
            heads: int,
            normalize_shift: bool,
            normalize_scale: bool,
            scale_init: float,
    ):
        super().__init__(radius_multiplier=radius_multiplier, max_radius=max_radius)

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.output_names = ['fc1', 'fc2']
        self.fc1 = IntervalLinear(
            self.input_size, self.hidden_dim,
            radius_multiplier=radius_multiplier, max_radius=max_radius,
            bias=bias, normalize_shift=normalize_shift, normalize_scale=normalize_scale,
            scale_init=scale_init
        )
        self.fc2 = IntervalLinear(
            self.hidden_dim, self.hidden_dim,
            radius_multiplier=radius_multiplier, max_radius=max_radius,
            bias=bias, normalize_shift=normalize_shift, normalize_scale=normalize_scale,
            scale_init=scale_init,
        )
        if heads > 1:
            # Incremental task, we don't have to use intervals
            self.last = nn.ModuleList([
                PointLinear(self.hidden_dim, self.output_classes) for _ in range(heads)
            ])
        else:
            self.last = nn.ModuleList([
                IntervalLinear(
                    self.hidden_dim,
                    self.output_classes,
                    radius_multiplier=radius_multiplier,
                    max_radius=max_radius,
                    bias=bias,
                    normalize_shift=normalize_shift,
                    normalize_scale=normalize_scale,
                    scale_init=scale_init,
                )])

    # MW: this is a modified function from avalanche
    def forward(self, x: torch.Tensor, task_labels: torch.Tensor) -> torch.Tensor:
        """ compute the output given the input `x` and task labels.

        :param x:
        :param task_labels: task labels for each sample.
        :return:
        """
        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels)
        else:
            unique_tasks = torch.unique(task_labels)

        full_out = {}
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item())

            if not full_out:
                for key, val in out_task.items():
                    full_out[key] = torch.empty(x.shape[0], *val.shape[1:],
                                                device=val.device).rename(None)
            for key, val in out_task.items():
                full_out[key][task_mask] = val.rename(None)

        for key, val in full_out.items():
            full_out[key] = val.refine_names("N", "bounds", "features")
        return full_out

    def forward_base(self, x: Tensor) -> dict[str, Tensor]:  # type: ignore
        x = x.refine_names("N", "C", "H", "W")  # type: ignore  # expected input shape
        x = x.rename(None)  # type: ignore  # drop names for unsupported operations
        x = x.flatten(1)  # (N, features)
        x = x.unflatten(1, (1, -1))  # type: ignore  # (N, bounds, features)
        x = x.tile((1, 3, 1))

        x = x.refine_names("N", "bounds", "features")  # type: ignore

        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))

        return {
            "fc1": fc1,
            "fc2": fc2,
        }

    def forward_single_task(self, x: Tensor, task_id: int) -> dict[str, Tensor]:
        # Get activations from the second-to-last layer
        activation_dict = self.forward_base(x)
        activation_dict["last"] = self.last[task_id](activation_dict["fc2"])
        return activation_dict

    @property
    def device(self):
        return self.fc1.weight.device


class IntervalVGG(IntervalModel):
    CFG = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    def __init__(self,
                 variant: str,
                 in_channels: int,
                 output_classes: int,
                 radius_multiplier: float,
                 max_radius: float,
                 bias: bool,
                 heads: int,
                 normalize_shift: bool,
                 normalize_scale: bool,
                 scale_init: float,
                 ):
        super().__init__(radius_multiplier=radius_multiplier, max_radius=max_radius)
        self.in_channels = in_channels
        self.output_classes = output_classes
        self.radius_multiplier = radius_multiplier
        self.max_radius = max_radius
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.bias = bias
        self.output_names = ['features', 'classifier']
        self.features = self.make_layers(self.CFG[variant])
        self.classifier = nn.Sequential(
            IntervalLinear(512, 4096, radius_multiplier=radius_multiplier, max_radius=max_radius, bias=True,
                           normalize_shift=normalize_shift, normalize_scale=normalize_scale, scale_init=scale_init),
            nn.ReLU(),
            IntervalDropout(),
            IntervalLinear(4096, 4096, radius_multiplier=radius_multiplier, max_radius=max_radius, bias=True,
                           normalize_shift=normalize_shift, normalize_scale=normalize_scale, scale_init=scale_init),
            nn.ReLU(),
            IntervalDropout()
        )
        if heads > 1:
            # Incremental task, we don't have to use intervals
            self.last = nn.ModuleList([
                PointLinear(4096, self.output_classes) for _ in range(heads)
            ])
        else:
            self.last = nn.ModuleList([
                IntervalLinear(
                    4096, output_classes, radius_multiplier=radius_multiplier, max_radius=max_radius,
                    bias=bias, normalize_shift=normalize_shift, normalize_scale=normalize_scale,
                    scale_init=scale_init
                )])

    def forward(self, x: Tensor, task_labels: torch.Tensor) -> Tensor:
        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels)
        else:
            unique_tasks = torch.unique(task_labels)

        full_out = {}
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item())

            if not full_out:
                for key, val in out_task.items():
                    full_out[key] = torch.empty(x.shape[0], *val.shape[1:],
                                                device=val.device).rename(None)
            for key, val in out_task.items():
                full_out[key][task_mask] = val.rename(None)

        for key, val in full_out.items():
            full_out[key] = val.refine_names("N", "bounds", ...)
        return full_out

    def forward_base(self, x: Tensor) -> dict[str, Tensor]:
        x = x.unsqueeze(1).tile(1, 3, 1, 1, 1)
        x = x.refine_names("N", "bounds", "C", "H", "W")  # type: ignore  # expected input shape
        features = x = self.features(x)
        x = x.rename(None).flatten(2).refine_names("N", "bounds", "features")
        x = self.classifier(x)
        return {
            "features": features,
            "classifier": x,
        }

    def forward_single_task(self, x: Tensor, task_id: int) -> dict[str, Tensor]:
        # Get activations from the second-to-last layer
        activation_dict = self.forward_base(x)
        activation_dict["last"] = self.last[task_id](activation_dict["classifier"])
        return activation_dict

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        input_channel = self.in_channels
        for l in cfg:
            if l == 'M':
                layers += [IntervalMaxPool2d(kernel_size=2, stride=2)]
                continue
            layers += [
                IntervalConv2d(input_channel, l, kernel_size=3, padding=1, radius_multiplier=self.radius_multiplier,
                               max_radius=self.max_radius, bias=self.bias)]
            if batch_norm:
                raise NotImplementedError()
            layers += [nn.ReLU()]
            input_channel = l
        return nn.Sequential(*layers)
