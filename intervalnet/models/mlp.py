import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter


class MLP(nn.Module):
    """
    Multi-layer perceptron.
    """

    def __init__(self, input_size: int, hidden_dim: int, output_classes: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes

        self.fc1 = nn.Linear(self.input_size, self.hidden_dim, bias=False)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.last = nn.Linear(self.hidden_dim, self.output_classes, bias=False)

    def forward(self, x: Tensor):  # type: ignore
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last(x)

        return x

    @property
    def device(self):
        return self.fc1.weight.device


class IntervalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.radius = Parameter(torch.empty((out_features, in_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        with torch.no_grad():
            self.radius.zero_()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = x.refine_names('N', 'bounds', 'features')  # type: ignore
        assert (x >= 0.0).all(), 'All input features must be non-negative.'

        x_lower, x_middle, x_upper = x.unbind('bounds')
        assert (x_lower <= x_middle).all(), 'Lower bound must be less than or equal to middle bound.'
        assert (x_middle <= x_upper).all(), 'Middle bound must be less than or equal to upper bound.'

        assert (self.radius >= 0).all(), 'All radii must be non-negative.'

        w_middle = self.weight
        w_lower = self.weight - self.radius
        w_upper = self.weight + self.radius

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


class IntervalMLP(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, output_classes: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes

        self.fc1 = IntervalLinear(self.input_size, self.hidden_dim)
        self.fc2 = IntervalLinear(self.hidden_dim, self.hidden_dim)
        self.last = IntervalLinear(self.hidden_dim, self.output_classes)

    def forward(self, x: Tensor):  # type: ignore
        x = x.refine_names('N', 'C', 'H', 'W')  # type: ignore  # expected input shape

        x = x.rename(None)  # type: ignore  # drop names for unsupported operations
        x = x.flatten(1)  # (N, features)
        x = x.unflatten(1, (1, -1))  # type: ignore  # (N, bounds, features)
        x = x.tile((1, 3, 1))

        x = x.refine_names('N', 'bounds', 'features')  # type: ignore

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last(x)

        return x

    @property
    def device(self):
        return self.fc1.weight.device
