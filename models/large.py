import torch
import torch.nn as nn
import torch.nn.functional as f
from interval.layers import LinearInterval, Conv2dInterval


class Large(nn.Module):

    def __init__(self, eps=0):
        super().__init__()
        self.conv1 = Conv2dInterval(3, 64, 3, 1, input_layer=True)
        self.conv2 = Conv2dInterval(64, 64, 3, 1)
        self.conv3 = Conv2dInterval(64, 128, 3, 2)
        self.conv4 = Conv2dInterval(128, 128, 3, 1)
        self.conv5 = Conv2dInterval(128, 128, 3, 1)
        self.fc1 = LinearInterval(128 * 9 * 9, 200)
        self.last = LinearInterval(200, 10)

        self.a = nn.Parameter(torch.zeros(7), requires_grad=True)
        self.e = None

        self.eps = eps
        self.bounds = None

    def save_bounds(self, x):
        self.bounds = x

    def calc_eps(self, r):
        exp = self.a.exp()
        self.e = r * exp / exp.sum()

    def print_eps(self):
        for c in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1):
            e1 = c.eps.detach()
            print(f"sum: {e1.sum()} - mean: {e1.mean()} - std: {e1.std()}")

        for name, layer in self.last.items():
            l = layer.eps.detach()
            print(f"last-{name} sum: {l.sum()} - mean: {l.mean()} - std: {l.std()}")

    def reset_importance(self):
        pass
        # self.conv1.reset_importance()
        # self.conv2.reset_importance()
        # self.conv3.reset_importance()
        # self.conv4.reset_importance()
        # self.conv5.reset_importance()
        # self.fc1.reset_importance()
        # for _, layer in self.last.items():
        #     layer.reset_importance()

    def set_eps(self, eps, trainable=False):
        if trainable:
            self.calc_eps(eps)

            self.conv1.calc_eps(self.e[0])
            self.conv2.calc_eps(self.e[1])
            self.conv3.calc_eps(self.e[2])
            self.conv4.calc_eps(self.e[3])
            self.conv5.calc_eps(self.e[4])
            self.fc1.calc_eps(self.e[5])
            for _, layer in self.last.items():
                layer.calc_eps(self.e[6])
        else:
            self.conv1.calc_eps(eps)
            self.conv2.calc_eps(eps)
            self.conv3.calc_eps(eps)
            self.conv4.calc_eps(eps)
            self.conv5.calc_eps(eps)
            self.fc1.calc_eps(eps)
            for _, layer in self.last.items():
                layer.calc_eps(eps)

    def features(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        self.save_bounds(x)
        return x

    def logits(self, x):
        return self.last(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return {k: v[:, :v.size(1) // 3] for k, v in x.items()}


def large():
    return Large()
