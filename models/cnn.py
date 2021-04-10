import torch
import torch.nn as nn
import torch.nn.functional as f
from interval.layers import LinearInterval, Conv2dInterval, MaxPool2dInterval, AvgPool2dInterval, IntervalDropout


class CNN(nn.Module):

    def __init__(self, in_channel=3, out_dim=10, pooling=nn.MaxPool2d):
        super().__init__()

        self.input = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.c1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            nn.Dropout(0.25)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            nn.Dropout(0.25)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=1),
            nn.Dropout(0.25)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*5*5, 256),
            nn.ReLU()
        )
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

        self.input = Conv2dInterval(in_channel, 32, kernel_size=3, stride=1, padding=1, input_layer=True)
        self.c1 = nn.Sequential(
            Conv2dInterval(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Conv2dInterval(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            IntervalDropout(0.25)
        )
        self.c2 = nn.Sequential(
            Conv2dInterval(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Conv2dInterval(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            IntervalDropout(0.25)
        )
        self.c3 = nn.Sequential(
            Conv2dInterval(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Conv2dInterval(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=1),
            IntervalDropout(0.25)
        )
        self.fc1 = nn.Sequential(
            LinearInterval(128 * 5 * 5, 256),
            nn.ReLU()
        )
        self.last = LinearInterval(256, out_dim)
        self.a = nn.Parameter(torch.zeros(9), requires_grad=True)
        self.e = None

        self.bounds = None

    def save_bounds(self, x):
        s = x.size(1) // 3
        self.bounds = x[:, s:2*s], x[:, 2*s:]

    def print_eps(self):
        e = self.input.eps.detach()
        print(f"sum: {e.sum()} - mean: {e.mean()} - std: {e.std()}")
        print(f"min: {e.min()} - max: {e.max()}")

        for c in (self.c1, self.c2, self.c3):
            e1 = c[0].eps.detach()
            e2 = c[2].eps.detach()
            print(f"sum: {e1.sum()} - mean: {e1.mean()} - std: {e1.std()}")
            print(f"min: {e1.min()} - max: {e1.max()}")
            print(f"sum: {e2.sum()} - mean: {e2.mean()} - std: {e2.std()}")
            print(f"min: {e2.min()} - max: {e2.max()}")

        e = self.fc1[0].eps.detach()
        print(f"sum: {e.sum()} - mean: {e.mean()} - std: {e.std()}")
        print(f"min: {e.min()} - max: {e.max()}")

        for name, layer in self.last.items():
            l = layer.eps.detach()
            print(f"last-{name} sum: {l.sum()} - mean: {l.mean()} - std: {l.std()}")
            print(f"min: {l.min()} - max: {l.max()}")

    def calc_eps(self, r):
        exp = self.a.exp()
        self.e = r * exp / exp.sum()

    def reset_importance(self):
        self.input.rest_importance()
        self.c1[0].rest_importance()
        self.c1[2].rest_importance()
        self.c2[0].rest_importance()
        self.c2[2].rest_importance()
        self.c3[0].rest_importance()
        self.c3[2].rest_importance()
        self.fc1[0].rest_importance()
        for _, layer in self.last.items():
            layer.rest_importance()

    def set_eps(self, eps, trainable=False):

        if trainable:
            self.calc_eps(eps)

            self.input.calc_eps(self.e[0])
            self.c1[0].calc_eps(self.e[1])
            self.c1[2].calc_eps(self.e[2])
            self.c2[0].calc_eps(self.e[3])
            self.c2[2].calc_eps(self.e[4])
            self.c3[0].calc_eps(self.e[5])
            self.c3[2].calc_eps(self.e[6])
            self.fc1[0].calc_eps(self.e[7])
            for _, layer in self.last.items():
                layer.calc_eps(self.e[8])
        else:
            self.input.calc_eps(eps)
            self.c1[0].calc_eps(eps)
            self.c1[2].calc_eps(eps)
            self.c2[0].calc_eps(eps)
            self.c2[2].calc_eps(eps)
            self.c3[0].calc_eps(eps)
            self.c3[2].calc_eps(eps)
            self.fc1[0].calc_eps(eps)
            for _, layer in self.last.items():
                layer.calc_eps(eps)

    def features(self, x):
        x = self.input(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        self.save_bounds(x)
        return x

    def logits(self, x):
        return self.last(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return {k: v[:, :v.size(1)//3] for k, v in x.items()}


def interval_cnn():
    return IntervalCNN()


def interval_cnn_avg():
    return IntervalCNN(pooling=AvgPool2dInterval)


if __name__ == '__main__':
    cnn = IntervalCNN()
    x = torch.randn(12, 3, 32, 32)
    cnn(x)
