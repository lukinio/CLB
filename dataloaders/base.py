import torchvision
from torchvision import transforms

from .wrapper import CacheClassLabel

def MNIST(dataroot, train_aug=False, normalize=True):
    print(f'creating MNIST with normalize: {normalize}')
    # Add padding to make 32x32
    val_transform_list = [
        transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
    ]
    if normalize:
        normalize_transform = transforms.Normalize(mean=(0.1000, ), std=(0.2752, ))  # for 32x32
        val_transform_list.append(normalize_transform)
        #normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28

    val_transform = transforms.Compose(val_transform_list)
    train_transform = val_transform
    if train_aug:
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ]
        if normalize:
            train_transform_list.append(normalize_transform)
        train_transform = transforms.Compose(train_transform_list)

    train_dataset = torchvision.datasets.MNIST(root=dataroot, train=True, download=True, transform=train_transform)
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(dataroot, train=False, transform=val_transform)
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR10(dataroot, train_aug=False, normalize=True):
    print(f'creating CIFAR10 with normalize: {normalize}')
    normalize_transform = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    val_transform_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        print(f'appending normalize transform')
        val_transform_list.append(normalize_transform)
    val_transform = transforms.Compose(val_transform_list)
    train_transform = val_transform
    if train_aug:
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if normalize:
            print(f'appending normalize transform')
            train_transform_list.append(normalize_transform)
        train_transform = transforms.Compose(train_transform_list)

    train_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=train_transform)
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=val_transform)
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR100(dataroot, train_aug=False, normalize=True):
    print(f'creating CIFAR100 with normalize: {normalize}')
    normalize_transform = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        val_transform_list.append(normalize_transform)
    val_transform = transforms.Compose(val_transform_list)
    train_transform = val_transform
    if train_aug:
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if normalize:
            train_transform_list.append(normalize_transform)
        train_transform = transforms.Compose(train_transform_list)

    train_dataset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=train_transform)
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=val_transform)
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset
