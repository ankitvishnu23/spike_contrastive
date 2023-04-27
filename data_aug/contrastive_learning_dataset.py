from fileinput import filename
import torch
import numpy as np
import os

from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from data_aug.view_generator import ContrastiveLearningViewGenerator, LabelViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data_aug.wf_data_augs import AmpJitter, Jitter, Collide, SmartNoise, ToWfTensor, PCA_Reproj
from typing import Any, Callable, Optional, Tuple

class WFDataset(Dataset):
    filename = "spikes_train.npy"
    multi_filename = "multichan_spikes_train.npy"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__()

        self.data: Any = []

        # now load the numpy array
        self.data = np.load(root + self.filename).astype('float32')
        print(self.data.shape)
        self.root = root
        self.transform = transform
        self.targets = np.array([[i for j in range(1200)] \
                                for i in range(10)]).reshape(-1).astype('long')
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Any :
        """
        Args:
            index (int): Index

        Returns:
            tensor: wf
        """
        wf = self.data[index].astype('float32')
        y = self.targets[index].astype('long')

        # doing this so that it is a tensor
        # wf = torch.from_numpy(wf)

        if self.transform is not None:
            wf = self.transform(wf)

        if self.target_transform is not None:
            y = self.target_transform(y)
            
        return wf, y


    def __len__(self) -> int:
        return len(self.data)


class WF_MultiChan_Dataset(Dataset):
    filename = "multichan_spikes_train.npy"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:

        super().__init__()

        self.data: Any = []

        # now load the numpy array
        self.data = np.load(root + self.filename)
        print(self.data.shape)
        self.root = root
        self.transform = transform
        self.targets = np.array([[i for j in range(1200)] \
                                for i in range(10)]).reshape(-1).astype('long')
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Any :
        """
        Args:
            index (int): Index

        Returns:
            tensor: wf
        """
        wf = self.data[index].astype('float32')
        y = self.targets[index].astype('long')
        # doing this so that it is a tensor
        # wf = torch.from_numpy(wf)

        if self.transform is not None:
            wf = self.transform(wf)
        
        if self.target_transform is not None:
            y = self.target_transform(y)

        return wf, y


    def __len__(self) -> int:
        return len(self.data)


class WFDataset_lab(Dataset):

    def __init__(
        self,
        root: str,
        multi_chan: bool = False,
        split: str = 'train',
        transform: Optional[Callable] = None,
        
    ) -> None:

        super().__init__()
        if split == 'train':
            if multi_chan == False:
                self.filename = "spikes_train.npy"
            else:
                self.filename = "multichan_spikes_train.npy"
            self.data = np.load(root + self.filename).astype('float32')
            self.targets = np.array([[i for j in range(1200)] \
                                for i in range(10)]).reshape(-1).astype('long')
        elif split == 'test':
            if multi_chan == False:
                self.filename = "spikes_test.npy"
            else:
                self.filename = "multichan_spikes_test.npy"
            self.data = np.load(root + self.filename).astype('float32')
            self.targets = np.array([[i for j in range(300)] \
                                for i in range(10)]).reshape(-1).astype('long')
            
        # self.data: Any = []

        # now load the numpy array
        print(self.data.shape)
        self.root = root
        self.transform = transform

    def __getitem__(self, index: int) -> Any :
        """
        Args:
            index (int): Index

        Returns:
            tensor: wf
        """
        wf = self.data[index].astype('float32')
        y = self.targets[index].astype('long')

        # doing this so that it is a tensor
        # wf = torch.from_numpy(wf)

        if self.transform is not None:
            wf = self.transform(wf)

        return wf, y

    def __len__(self) -> int:
        return len(self.data)

class ContrastiveLearningDataset:
    def __init__(self, root_folder, lat_dim, multi_chan):
        self.root_folder = root_folder
        self.lat_dim = lat_dim
        self.multi_chan = multi_chan
    

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
    def get_wf_pipeline_transform(self, temp_cov_fn, spatial_cov_fn, noise_scale):
        temporal_cov = np.load(os.path.join(self.root_folder, temp_cov_fn))
        spatial_cov = np.load(os.path.join(self.root_folder, spatial_cov_fn))
        """Return a set of data augmentation transformations on waveforms."""
        data_transforms = transforms.Compose([
                                            transforms.RandomApply([AmpJitter()], p=0.7),
                                              transforms.RandomApply([Jitter()], p=0.6),
                                            #   transforms.RandomApply([PCA_Reproj(root_folder=self.root_folder)], p=0.4),
                                              transforms.RandomApply([SmartNoise(self.root_folder, temporal_cov, spatial_cov, noise_scale)], p=0.5),
                                              transforms.RandomApply([Collide(self.root_folder)], p=0.4),
                                              ToWfTensor()])
        
        return data_transforms

    @staticmethod
    def get_pca_transform(self):
        data_transforms = transforms.Compose([PCA_Reproj(root_folder=self.root_folder, pca_dim=self.lat_dim),
                                              ToWfTensor()])
        
        return data_transforms

    def get_dataset(self, name, n_views, noise_scale=1.0):
        temp_cov_fn = 'temporal_cov_example.npy'    
        spatial_cov_fn = 'spatial_cov_example.npy'
        if self.multi_chan:
            name = name + '_multichan'
        valid_datasets = {'wfs': lambda: WFDataset(self.root_folder,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_wf_pipeline_transform(self, temp_cov_fn,
                                                                  spatial_cov_fn,
                                                                #   noise_scale), self.get_pca_transform(self),
                                                                  noise_scale), None,
                                                                  n_views),
                                                              target_transform=LabelViewGenerator()),
                          'wfs_multichan': lambda: WF_MultiChan_Dataset(self.root_folder,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_wf_pipeline_transform(self, temp_cov_fn,
                                                                  spatial_cov_fn,
                                                                #   noise_scale), self.get_pca_transform(self),
                                                                  noise_scale), None,
                                                                  n_views),
                                                              target_transform=LabelViewGenerator()),
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
    


