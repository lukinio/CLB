import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, input_size: int, hidden_dim: int, output_classes: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes

        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.last = nn.Linear(self.hidden_dim, self.output_classes)

    def forward(self, x: Tensor):  # type: ignore
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last(x)

        return x

    @property
    def device(self):
        return self.fc1.weight.device
