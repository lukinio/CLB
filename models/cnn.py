import torch
import torch.nn as nn
import torch.nn.functional as f
from interval.layers import (AvgPool2dInterval, Conv2dInterval, IntervalBias,
                             IntervalDropout, LinearInterval,
                             MaxPool2dInterval)


class CNN(nn.Module):
    def __init__(self, in_channel=3, out_dim=10, pooling=nn.MaxPool2d):
        super().__init__()

        self.c1 = nn.Sequential(nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
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
    def __init__(self, in_channel=3, out_dim=2, pooling=MaxPool2dInterval):
        super(IntervalCNN, self).__init__()

        self.c1 = nn.Sequential(
            Conv2dInterval(in_channel, 32, kernel_size=3, stride=1, padding=1, input_layer=True),
            # IntervalBias(32),
            nn.ReLU(),
            Conv2dInterval(32, 32, kernel_size=3, stride=1, padding=1),
            # IntervalBias(32),
            nn.ReLU(),
            Conv2dInterval(32, 64, kernel_size=3, stride=1, padding=1),
            # IntervalBias(64),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            IntervalDropout(0.25))
        self.c2 = nn.Sequential(
            Conv2dInterval(64, 64, kernel_size=3, stride=1, padding=1),
            # IntervalBias(64),
            nn.ReLU(),
            Conv2dInterval(64, 128, kernel_size=3, stride=1, padding=1),
            # IntervalBias(128),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            IntervalDropout(0.25))
        self.c3 = nn.Sequential(
            Conv2dInterval(128, 128, kernel_size=3, stride=1, padding=1),
            # IntervalBias(128),
            nn.ReLU(),
            Conv2dInterval(128, 128, kernel_size=3, stride=1, padding=1),
            # IntervalBias(128),
            nn.ReLU(),
            pooling(2, stride=2, padding=1),
            IntervalDropout(0.25))
        self.fc1 = nn.Sequential(
            LinearInterval(128 * 5 * 5, 256),
            # IntervalBias(256),
            nn.ReLU())
        self.last = nn.Sequential(LinearInterval(256, out_dim),
                                  # IntervalBias(out_dim),
                                  )

        sum_numel = 0
        self.numels = []
        for m in self.modules():
            if isinstance(m, (Conv2dInterval, LinearInterval, IntervalBias)):
                numwei = m.weight.numel()
                # print(f'type(m): {type(m)} numwei: {numwei}')
                sum_numel += numwei
                self.numels.append(numwei)
        self.importances = nn.Parameter(torch.zeros(sum_numel, requires_grad=True, device=self.fc1[0].weight.device))

    def importances_to_eps(self, eps_scaler):
        base_eps = torch.softmax(self.importances, dim=0)
        eps = base_eps * eps_scaler
        split_eps = torch.split(eps, self.numels)
        i = 0
        for m in self.modules():
            if isinstance(m, (Conv2dInterval, LinearInterval, IntervalBias)):
                m.eps = split_eps[i].view_as(m.weight)
                i += 1

    def print_eps_stats(self, head="All"):
        eps_sum = 0
        for i, m in enumerate(self.modules()):
            if isinstance(m, (Conv2dInterval, LinearInterval, IntervalBias)):
                eps = m.eps.detach()
                print(f'module {i}: {type(m)}')
                print(f'  sum: {eps.sum()} mean: {eps.mean()} std: {eps.std()}')
                print(f'  min: {eps.min()} max: {eps.max()} numel: {eps.numel()}')
                eps_sum += eps.sum().item()
        print(f'Total eps sum: {eps_sum}')
        

    def reset_importances(self):
        self.importances = nn.Parameter(torch.zeros_like(self.importances))

    def features(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = self.features(x)
        answers = {k: self.last[k](x) for k, v in self.last.items()}
        return {k: v for k, v in answers.items()}


def interval_cnn():
    return IntervalCNN()


def interval_cnn_avg():
    return IntervalCNN(pooling=AvgPool2dInterval)


if __name__ == '__main__':
    cnn = IntervalCNN()

    x = torch.randn(12, 3, 32, 32)
    print(f"before: {cnn.features_loss_term}")
    cnn(x)
    print(f"after: {cnn.features_loss_term}")
