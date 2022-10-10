from fileinput import filename
import torch
import numpy as np

from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data_aug.wf_data_augs import AmpJitter, Jitter, Collide, Noise, SmartNoise, ToWfTensor
from typing import Any, Callable, Optional, Tuple

class WFDataset(Dataset):
    filename = "kilo_hptp_mcs.npy"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None
    ) -> None:

        super().__init__()

        self.data: Any = []

        # now load the numpy array
        self.data = np.load(root + self.filename)
        self.root = root
        self.transform = transform

    def __getitem__(self, index: int) -> Any :
        """
        Args:
            index (int): Index

        Returns:
            tensor: wf
        """
        wf = self.data[index]

        # doing this so that it is a tensor
        # wf = torch.from_numpy(wf)

        if self.transform is not None:
            wf = self.transform(wf)

        return wf


    def __len__(self) -> int:
        return len(self.data)


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder
    

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms
    
    @staticmethod
    def get_wf_pipeline_transform(self, temp_cov_fn):
        temporal_cov = np.load(self.root_folder + temp_cov_fn)
        print(temporal_cov.shape)
        """Return a set of data augmentation transformations on waveforms."""
        data_transforms = transforms.Compose([transforms.RandomApply([AmpJitter()], p=0.7),
                                              transforms.RandomApply([Jitter()], p=0.6),
                                              transforms.RandomApply([SmartNoise(temporal_cov)], p=0.5),
                                              transforms.RandomApply([Collide()], p=0.4),
                                              ToWfTensor()])
        
        return data_transforms

    def get_dataset(self, name, n_views):
        temp_cov_fn = 'temporal_cov_example.npy'
        valid_datasets = {'wfs': lambda: WFDataset(self.root_folder,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_wf_pipeline_transform(self, temp_cov_fn),
                                                                  n_views)),
                          'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
