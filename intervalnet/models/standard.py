import torch.nn as nn
import torch.nn.functional as F
from avalanche.models import MultiTaskModule
from torch import Tensor


class MLP(MultiTaskModule):
    """Multi-layer perceptron."""

    def __init__(self, input_size: int, hidden_dim: int, output_classes: int, heads: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes

        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.last = nn.ModuleList(nn.Linear(self.hidden_dim, self.output_classes) for _ in range(heads))

    def forward_single_task(self, x: Tensor, task_id: int):  # type: ignore
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last[task_id](x)

        return x

    @property
    def device(self):
        return self.fc1.weight.device


class CNN(MultiTaskModule):
    def __init__(self, in_channels, out_dim, heads: int):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.MaxPool2d(2, stride=2, padding=0), nn.Dropout(0.25))
        self.c2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.MaxPool2d(2, stride=2, padding=0), nn.Dropout(0.25))
        self.c3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.MaxPool2d(2, stride=2, padding=1), nn.Dropout(0.25))
        self.fc1 = nn.Sequential(nn.Linear(128 * 5 * 5, 256), nn.ReLU())
        self.last = nn.ModuleList(nn.Linear(256, out_dim) for _ in range(heads))

    def forward_single_task(self, x: Tensor, task_id: int):  # type: ignore
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.last[task_id](x)
        return x

    @property
    def device(self):
        return self.fc1.weight.device
