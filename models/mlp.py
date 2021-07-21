import torch
import torch.nn as nn
import torch.nn.functional as f
from interval.layers import IntervalLayerWithParameters, LinearInterval
from math import sqrt


class IntervalMLP(nn.Module):
    def __init__(self, task_output_space=None, in_channel=1, img_sz=32, hidden_dim=256):
        super(IntervalMLP, self).__init__()
        self.in_dim = in_channel * img_sz * img_sz
        self.fc1 = LinearInterval(self.in_dim, hidden_dim, input_layer=True)
        self.fc2 = LinearInterval(hidden_dim, hidden_dim)
        # Subject to be replaced dependent on task
        self.hidden_dim = hidden_dim
        self.last = None
        self.reset_heads(task_output_space)
        self.reset_importances()

    def reset_heads(self, task_output_space):
        self.last = nn.ModuleDict()
        for task, out_dim in task_output_space.items():
            # self.last[task] = nn.Sequential(LinearInterval(self.hidden_dim, out_dim), IntervalBias(out_dim))
            self.last[task] = nn.Sequential(LinearInterval(self.hidden_dim, out_dim), )

    def reset_importances(self):
        sum_numel = 0
        self.numels = []
        for m in self.modules():
            if isinstance(m, IntervalLayerWithParameters):
                numwei = m.weight.numel()
                # print(f'type(m): {type(m)} numwei: {numwei}')
                sum_numel += numwei
                self.numels.append(numwei)
        self.importances = nn.Parameter(torch.zeros((sum_numel, 1), requires_grad=False, device=self.fc1.weight.device))
        nn.init.kaiming_uniform_(self.importances, a=sqrt(5))
        self.importances.data = self.importances[:, 0]

    def importances_to_eps(self, eps_scaler, mode='sum'):
        assert mode in ['sum', 'product']
        eps_scaler = 1140800
        base_eps = torch.softmax(self.importances, dim=0)
        if mode == 'product':
            eps = torch.pow(eps_scaler, base_eps)
        else:
            eps = base_eps * eps_scaler
        split_eps = torch.split(eps, self.numels)
        i = 0
        for m in self.modules():
            if isinstance(m, IntervalLayerWithParameters):
                m.eps = split_eps[i].view_as(m.weight)
                i += 1

    def print_eps_stats(self, head="All"):
        eps_sum = 0
        for i, m in enumerate(self.modules()):
            if isinstance(m, IntervalLayerWithParameters):
                eps = m.eps.detach()
                print(f'module {i}: {type(m)}')
                print(f'  sum: {eps.sum()} mean: {eps.mean()} std: {eps.std()}')
                print(f'  min: {eps.min()} max: {eps.max()} numel: {eps.numel()}')
                eps_sum += eps.sum().item()
        print(f'Total eps sum: {eps_sum}')

    def features(self, x):
        x = x.view(-1, self.in_dim)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return x

    def forward(self, x):
        x = self.features(x)
        answers = {k: self.last[k](x) for k, v in self.last.items()}
        return {k: v for k, v in answers.items()}


def interval_mlp400(task_output_space):
    return IntervalMLP(task_output_space, hidden_dim=400)


class MLP(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        super(MLP, self).__init__()
        self.in_dim = in_channel * img_sz * img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1, self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def MLP100(task_output_space):
    return MLP(task_output_space=task_output_space, hidden_dim=100)


def MLP400(task_output_space):
    return MLP(task_output_space=task_output_space, hidden_dim=400)


def MLP1000(task_output_space):
    return MLP(task_output_space=task_output_space, hidden_dim=1000)


def MLP2000(task_output_space):
    return MLP(task_output_space=task_output_space, hidden_dim=2000)


def MLP5000(task_output_space):
    return MLP(task_output_space=task_output_space, hidden_dim=5000)
