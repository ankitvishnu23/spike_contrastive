import os
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from PIL import Image, ImageOps, ImageFilter

from utils import GaussianBlur, Solarization

def build_dataset(args):
    if args.mask_mode == 'topk' or args.mask_mode == 'topk_sum' or args.mask_mode == 'topk_agg_sum':
        dataset = TopKDataset(args.dataset, args.data / 'train', args.topk_path, args.topk, Transform(args))
    elif args.mask_mode == 'weight_anchor_logits' or args.mask_mode == 'weight_class_logits':
        dataset = LogitsDataset(args.dataset, args.data / 'train', args.topk_path, Transform(args))
    else:
        if args.dataset == 'imagenet':
            dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform(args))
        elif args.dataset == 'cifar100':
            dataset = torchvision.datasets.CIFAR100(args.data, transform=Transform(args), train=True)

    return dataset

class TopKDataset(Dataset):
    def __init__(self, dataset, data_dir, topk_path, topk, transform=None):
        if dataset == 'imagenet':
            self.dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
        elif dataset == 'cifar100':
            self.dataset = torchvision.datasets.CIFAR100(data_dir, transform=transform, train=True)
        self.topk = topk

        if os.path.isfile(topk_path):
            with open(topk_path, 'rb') as f:
                self.topk_dict = pickle.load(f)
        else:
            raise Exception('Pickle load failed')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (y1, y2), label = self.dataset[idx]

        topk_labels = self.topk_dict[idx].long()

        if len(topk_labels) < 10:
            pad = torch.empty(10-len(topk_labels)).fill_(-1.0).long()
            topk_labels = torch.cat([topk_labels, pad])

        return (y1, y2), (label, topk_labels[:self.topk])


class LogitsDataset(Dataset):
    def __init__(self, dataset, data_dir, logits_path, transform=None):
        if dataset == 'imagenet':
            self.dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
        elif dataset == 'cifar100':
            self.dataset = torchvision.datasets.CIFAR100(data_dir, transform=transform, train=True)

        self.logits = np.memmap(logits_path, dtype='float32', mode='r', shape=(len(self.dataset),1000))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (y1, y2), label = self.dataset[idx]

        logits = torch.from_numpy(self.logits[idx])

        return (y1, y2), (label, logits)

class IdxDataset(Dataset):
    """Paired ImageFolder (for both ImageNet and StylizedImageNet, in that order)"""
    def __init__(self, dataset, root, transform=None, target_transform=None, is_valid_file=None, train=True):
        if dataset == 'cifar10':
            self.data = CIFAR10(root=root,
                                transform=transform,
                                train=train,
                                download=True)
        elif dataset == 'cifar100':
            self.data = CIFAR100(root=root,
                                 transform=transform,
                                 train=train,
                                 download=True)
        elif dataset == 'imagenet' or dataset == 'path':
            self.data = ImageFolder(root=root, transform=transform)
        else:
            raise ValueError(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]

        return x, y, idx

class Transform:
    def __init__(self, args):
        self.args = args
        if args.dataset == 'imagenet':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            if args.weak_aug:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                saturation=0.2, hue=0.1)],
                        p=1.0
                    ),
                    transforms.RandomGrayscale(p=0.5),
                    GaussianBlur(p=1.0),
                    Solarization(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])

                self.transform_prime = transforms.Compose([
                    transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform_prime = transforms.Compose([
                    transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                saturation=0.2, hue=0.1)],
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(p=0.1),
                    Solarization(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])

            self.transform_rotation = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=(args.scale[0], args.scale[1])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

        elif args.dataset == 'cifar100':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761])
            ])

            self.transform_prime = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761])
            ])

            self.transform_rotation = self.transform_prime

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        y3 = self.transform_rotation(x)

        return y1, y2, y3
