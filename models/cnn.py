import torch
import torch.nn as nn
import torch.nn.functional as f
from interval.layers import (AvgPool2dInterval, Conv2dInterval, IntervalBias, IntervalDropout, LinearInterval,
                             MaxPool2dInterval)


class CNN(nn.Module):
    def __init__(self, in_channel=3, out_dim=10, pooling=nn.MaxPool2d):
        super().__init__()

        self.input = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.c1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                pooling(2, stride=2, padding=0), nn.Dropout(0.25))
        self.c2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                pooling(2, stride=2, padding=0), nn.Dropout(0.25))
        self.c3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                pooling(2, stride=2, padding=1), nn.Dropout(0.25))
        self.fc1 = nn.Sequential(nn.Linear(128 * 5 * 5, 256), nn.ReLU())
        self.last = nn.Linear(256, out_dim)

    def features(self, x):
        x = self.input(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def logits(self, x):
        return self.last(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def cnn():
    return CNN()


def cnn_avg():
    return CNN(pooling=nn.AvgPool2d)


class IntervalCNN(nn.Module):
    def __init__(self, in_channel=3, out_dim=10, pooling=MaxPool2dInterval):
        super(IntervalCNN, self).__init__()

        self.input = nn.Sequential(
            Conv2dInterval(in_channel, 32, kernel_size=3, stride=1, padding=1, input_layer=True),
            IntervalBias(32),
        )
        self.c1 = nn.Sequential(Conv2dInterval(32, 32, kernel_size=3, stride=1, padding=1), IntervalBias(32), nn.ReLU(),
                                Conv2dInterval(32, 64, kernel_size=3, stride=1, padding=1), IntervalBias(64), nn.ReLU(),
                                pooling(2, stride=2, padding=0), IntervalDropout(0.25))
        self.c2 = nn.Sequential(Conv2dInterval(64, 64, kernel_size=3, stride=1, padding=1), IntervalBias(64), nn.ReLU(),
                                Conv2dInterval(64, 128, kernel_size=3, stride=1, padding=1), IntervalBias(128),
                                nn.ReLU(), pooling(2, stride=2, padding=0), IntervalDropout(0.25))
        self.c3 = nn.Sequential(Conv2dInterval(128, 128, kernel_size=3, stride=1, padding=1), IntervalBias(128),
                                nn.ReLU(), Conv2dInterval(128, 128, kernel_size=3, stride=1, padding=1),
                                IntervalBias(128), nn.ReLU(), pooling(2, stride=2, padding=1), IntervalDropout(0.25))
        self.fc1 = nn.Sequential(LinearInterval(128 * 5 * 5, 256), IntervalBias(256), nn.ReLU())
        self.last = nn.Sequential(
            LinearInterval(256, out_dim),
            IntervalBias(out_dim),
        )
        # self.a = nn.Parameter(torch.Tensor([0, 0, 0, 0, 0, 0, 0, 10, 0]), requires_grad=True)
        self.a = nn.Parameter(torch.zeros(18), requires_grad=True)
        self.e = torch.zeros(18)
        self.bounds = None

    def save_bounds(self, x):
        self.bounds = x

    def calc_eps(self, r):
        exp = self.a.exp()
        self.e = r * exp / exp.sum()

    def print_eps(self, head="All"):
        for m in self.modules():
            if isinstance(m, (Conv2dInterval, LinearInterval, IntervalBias)):
                e = m.eps.detach()
                print(f"sum: {e.sum()} - mean: {e.mean()} - std: {e.std()}")
                print(f" * min {e.min()}, max: {e.max()}")
        print(f"eps: {self.e}")

    def reset_importance(self):
        for m in self.modules():
            if isinstance(m, (Conv2dInterval, LinearInterval, IntervalBias)):
                m.rest_importance()

    def set_eps(self, eps, trainable=False, head="All"):
        if trainable:
            self.calc_eps(eps)
        else:
            self.e[:] = eps
        i = 0
        for m in self.modules():
            if isinstance(m, (Conv2dInterval, LinearInterval, IntervalBias)):
                m.calc_eps(self.e[i])
                i += 1

    def features(self, x):
        x = self.input(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # TODO remove statefulness
        self.save_bounds(x)
        return x

    def forward(self, x):
        x = self.features(x)
        answers = {k: self.last[k](x) for k, v in self.last.items()}
        return {k: v[:, :v.size(1) // 3] for k, v in answers.items()}


def interval_cnn():
    return IntervalCNN()


def interval_cnn_avg():
    return IntervalCNN(pooling=AvgPool2dInterval)


if __name__ == '__main__':
    cnn = IntervalCNN()
    x = torch.randn(12, 3, 32, 32)
    cnn(x)
