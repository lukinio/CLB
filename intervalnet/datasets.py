import os
import warnings

import torch
from torch.utils.data import Subset, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST


def mnist():
    """
    MNIST dataset.
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mnist_train = MNIST(root=os.getenv('DATA_DIR', ''), train=True, download=True)
        mnist_test = MNIST(root=os.getenv('DATA_DIR', ''), train=False, download=True)

    mnist_transforms = transforms.Compose([
        #transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        #transforms.Normalize(mean=(0.1000,), std=(0.2752,)),
    ])

    return mnist_train, mnist_test, mnist_transforms

def mnist2():
    """
    MNIST dataset.
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mnist_train = MNIST(root=os.getenv('DATA_DIR', ''), train=True, download=True)
        mnist_test = MNIST(root=os.getenv('DATA_DIR', ''), train=False, download=True)

        # Filter out all classes except [0, 1, 2, 3]
        train_indices = [idx for idx, (_, y) in enumerate(mnist_train) if y <= 3]
        mnist_train = Subset(mnist_train, train_indices)

        test_indices = [idx for idx, (_, y) in enumerate(mnist_test) if y <= 3]
        mnist_test = Subset(mnist_test, test_indices)

    mnist_transforms = transforms.Compose([
        #transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        #transforms.Normalize(mean=(0.1000,), std=(0.2752,)),
    ])

    return mnist_train, mnist_test, mnist_transforms

def f1(X: torch.Tensor):
    return torch.sin(X - 10)

def f2(X: torch.Tensor):
    return (X - 10) ** 2

def two_1d_functions():
    first_domain = torch.linspace(0, 10, steps=100000)
    first_domain = first_domain.view(-1, 1)
    first_values = f1(first_domain)

    second_domain = torch.linspace(10, 20, steps=100000)
    second_domain = second_domain.view(-1, 1)
    second_values = f2(second_domain)

    first_dataset = TensorDataset(first_domain, first_values)
    second_dataset = TensorDataset(second_domain, second_values)
    full_dataset = [first_dataset, second_dataset]

    return full_dataset, full_dataset, None
