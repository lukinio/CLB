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


class VGG(nn.Module):
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
                 heads: int):
        super().__init__()
        self.features = self.make_layers(self.CFG[variant], in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.last = nn.ModuleList(nn.Linear(4096, output_classes) for _ in range(heads))

    def forward(self, x: Tensor, task_id: int) -> Tensor:
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        x = self.last[task_id](x)
        return output

    @staticmethod
    def make_layers(cfg, in_channels, batch_norm=False):
        layers = []
        input_channel = in_channels
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue
            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            if batch_norm:
                layers += [nn.BatchNorm2d(l)]
            layers += [nn.ReLU(inplace=True)]
            input_channel = l
        return nn.Sequential(*layers)
