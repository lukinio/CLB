import torch
import torch.nn as nn
import torch.nn.functional as f


def split_activation(x):
    s = x.size(1) // 3
    mid = x[:, :s]
    low = x[:, s:2 * s]
    upp = x[:, 2 * s:]
    return mid, low, upp


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    return flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)


ATOL = 1e-10


class AvgPool2dInterval(nn.AvgPool2d):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode=False,
                 count_include_pad=True,
                 divisor_override=None):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x_middle, x_lower, x_upper = split_activation(x)
        mid = super().forward(x_middle)
        lower = super().forward(x_lower)
        upper = super().forward(x_upper)
        return torch.cat((mid, lower, upper), dim=1)


class MaxPool2dInterval(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPool2dInterval, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        x_middle, x_lower, x_upper = split_activation(x)
        mid = super().forward(x_middle)
        lower = super().forward(x_lower)
        upper = super().forward(x_upper)

        # mid, ini = super().forward(x_middle)
        # lower = retrieve_elements_from_indices(x_lower, ini)
        # upper = retrieve_elements_from_indices(x_upper, ini)
        return torch.cat((mid, lower, upper), dim=1)


class IntervalDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.scale = 1. / (1 - self.p)

    def forward(self, x):
        if self.training:
            x_mid, x_low, x_upp = split_activation(x.clone())
            mask = torch.bernoulli(self.p * torch.ones_like(x_mid)).long()
            x_mid[mask == 1] = 0.
            x_low[mask == 1] = 0.
            x_upp[mask == 1] = 0.
            return torch.cat((x_mid, x_low, x_upp), dim=1) * self.scale
        else:
            return x


def check_intervals(middle, lower, upper):
    # assert torch.logical_or(lower <= upper, torch.isclose(lower, upper, atol=ATOL)).all(), f'diff:\n{lower - upper} '
    # assert torch.logical_or(lower <= middle, torch.isclose(lower, middle, atol=ATOL)).all(), f'diff:\n{lower - middle}'
    # assert torch.logical_or(middle <= upper, torch.isclose(middle, upper, atol=ATOL)).all(), f'diff:\n{middle - upper}'
    assert (lower <= upper).all(), f'diff:\n{lower - upper} '
    assert (lower <= middle).all(), f'diff:\n{lower - middle}'
    assert (middle <= upper).all(), f'diff:\n{middle - upper}'


class LinearInterval(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, input_layer=False):
        super().__init__(in_features, out_features, bias)
        # self.eps = torch.zeros_like(self.weight.data, requires_grad=True)
        self.eps = 0
        self.input_layer = input_layer

    def forward(self, x):
        assert (x >= 0.0).all(), f'x: {x}'
        if isinstance(self.eps, torch.Tensor):
            assert (self.eps >= 0.0).all()
        if self.input_layer:
            x_middle, x_lower, x_upper = x, x, x
        else:
            x_middle, x_lower, x_upper = split_activation(x)

        w_middle = self.weight
        w_lower = w_middle - self.eps
        w_upper = w_middle + self.eps

        w_lower_pos = (w_lower).clamp(min=0).t()
        w_lower_neg = (w_lower).clamp(max=0).t()
        w_upper_pos = (w_upper).clamp(min=0).t()
        w_upper_neg = (w_upper).clamp(max=0).t()

        l_lp = x_lower @ w_lower_pos
        u_ln = x_upper @ w_lower_neg
        u_up = x_upper @ w_upper_pos
        l_un = x_lower @ w_upper_neg

        lower = l_lp + u_ln
        upper = u_up + l_un

        w_middle_pos = (w_middle).clamp(min=0).t()
        w_middle_neg = (w_middle).clamp(max=0).t()
        m_mp = x_middle @ w_middle_pos
        m_mn = x_middle @ w_middle_neg
        middle = m_mp + m_mn

        check_intervals(middle, lower, upper)
        return torch.cat((middle, lower, upper), dim=1)


class Conv2dInterval(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 input_layer=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.eps = 0
        self.input_layer = input_layer

    def forward(self, x):
        assert (x >= 0.0).all(), f'x: {x}'
        if isinstance(self.eps, torch.Tensor):
            assert (self.eps >= 0.0).all()
        if self.input_layer:
            x_middle, x_lower, x_upper = x, x, x
        else:
            x_middle, x_lower, x_upper = split_activation(x)

        w_middle = self.weight
        w_lower = w_middle - self.eps
        w_upper = w_middle + self.eps

        w_lower_pos = (w_lower).clamp(min=0)
        w_lower_neg = (w_lower).clamp(max=0)
        w_upper_pos = (w_upper).clamp(min=0)
        w_upper_neg = (w_upper).clamp(max=0)

        l_lp = f.conv2d(x_lower, w_lower_pos, None, self.stride, self.padding, self.dilation, self.groups)
        u_ln = f.conv2d(x_upper, w_lower_neg, None, self.stride, self.padding, self.dilation, self.groups)
        u_up = f.conv2d(x_upper, w_upper_pos, None, self.stride, self.padding, self.dilation, self.groups)
        l_un = f.conv2d(x_lower, w_upper_neg, None, self.stride, self.padding, self.dilation, self.groups)

        lower = l_lp + u_ln
        upper = u_up + l_un

        w_middle_pos = w_middle.clamp(min=0)
        w_middle_neg = w_middle.clamp(max=0)
        m_mp = f.conv2d(x_middle, w_middle_pos, None, self.stride, self.padding, self.dilation, self.groups)
        m_mn = f.conv2d(x_middle, w_middle_neg, None, self.stride, self.padding, self.dilation, self.groups)
        middle = m_mp + m_mn

        check_intervals(middle, lower, upper)
        return torch.cat((middle, lower, upper), dim=1)


class IntervalBias(nn.Module):
    def __init__(self, bias_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(bias_size), requires_grad=True)
        self.eps = torch.zeros_like(self.weight.data, requires_grad=True)

    def forward(self, x):
        if isinstance(self.eps, torch.Tensor):
            assert (self.eps >= 0.0).all()
        x_mid, x_low, x_upp = split_activation(x.clone())
        if x.dim() == 4:
            b_lower = (self.weight - self.eps).view(1, -1, 1, 1)
            b_middle = (self.weight).view(1, -1, 1, 1)
            b_upper = (self.weight + self.eps).view(1, -1, 1, 1)
        elif x.dim() == 2:
            b_lower = (self.weight - self.eps).view(1, -1)
            b_middle = (self.weight).view(1, -1)
            b_upper = (self.weight + self.eps).view(1, -1)
        lower = x_low + b_lower
        middle = x_mid + b_middle
        upper = x_upp + b_upper

        check_intervals(middle, lower, upper)
        return torch.cat((middle, lower, upper), dim=1)


if __name__ == '__main__':
    # li = LinearInterval(5, 3)
    # for n, p in li.named_parameters():
    #     print(f"name: {n}")
    #     print(n, p)

    pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    unpool = nn.MaxUnpool2d(2, stride=2)
    input = torch.tensor([[[[1., 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    input1 = torch.tensor([[[[1., 2, 3, 4], [5, 0, 7, 0], [9, 10, 11, 12], [13, 0, 15, 0]]]])

    print(input.size())
    print(input)
    output, ini = pool(input)
    print(f"output: {output.size()}, ini: {ini.size()}")
    print(output)
    print(ini)
    print()
    fini = torch.flatten(ini, 2)
    ft = torch.flatten(input1, 2)
    print(ft.size())
    print(ft[:, :, fini].view(output.size()))

    o = retrieve_elements_from_indices(input1, ini)
    print(o)

    # print(f"fini: {ini.size()}")
    # print(unpool(output, ini) == input)
    # print(input1[unpool(output, ini) == input])
    # mask = (unpool(output, ini) == input).long()
    # print(input1.gather(dim=2, index=mask))
