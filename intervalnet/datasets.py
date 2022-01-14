import os
import warnings

import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.mnist import MNIST


def mnist():
    """MNIST dataset."""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mnist_train = MNIST(root=os.getenv('DATA_DIR', ''), train=True, download=True)
        mnist_test = MNIST(root=os.getenv('DATA_DIR', ''), train=False, download=True)
    mnist_train_transform = transforms.Compose([
        # transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
    ])
    mnist_eval_transform = transforms.Compose([        # transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
    ])
    return mnist_train, mnist_test, mnist_train_transform, mnist_eval_transform


def cifar100():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cifar_train = CIFAR100(root=os.getenv('DATA_DIR', ''), train=True, download=True)
        cifar_test = CIFAR100(root=os.getenv('DATA_DIR', ''), train=False, download=True)
    cifar_train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
    ])
    cifar_eval_transform = transforms.Compose([
        transforms.ToTensor(),

    ])
    return cifar_train, cifar_test, cifar_train_transform, cifar_eval_transform


def cifar10():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cifar_train = CIFAR10(root=os.getenv('DATA_DIR', ''), train=True, download=True)
        cifar_test = CIFAR10(root=os.getenv('DATA_DIR', ''), train=False, download=True)
    cifar_train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
    ])
    cifar_eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return cifar_train, cifar_test, cifar_train_transform, cifar_eval_transform
