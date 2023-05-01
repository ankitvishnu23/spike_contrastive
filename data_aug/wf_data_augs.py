from __future__ import print_function, division
from array import array
import os
import math
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
# from torchaudio.transforms import Resample

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class AmpJitter(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, lo=0.9, hi=1.1):
        self.lo = lo
        self.hi = hi

    def __call__(self, sample):
        if len(sample) == 2:
            wf, chan_nums = sample
        else:
            wf = sample

        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]
        
        amp_jit_value = np.random.uniform(self.lo, self.hi)
        amp_jit = np.array([amp_jit_value for i in range(n_chans)])

        wf = wf * amp_jit[:, None]
    
        return [wf, chan_nums]

class GaussianNoise(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __call__(self, sample):
        if len(sample) == 2:
            wf, chan_nums = sample
        else:
            wf = sample
        
        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]
        w = wf.shape[1]
        
        noise_wf = np.random.normal(0, 1, (n_chans, w))
        wf = np.add(wf, noise_wf)

        return [wf, chan_nums]

class SmartNoise(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    root_folder = '/Users/ankit/Documents/PaninskiLab/contrastive_spikes/DY016/'
    temporal_name = 'temporal_cov_example.npy'
    spatial_name = 'spatial_cov_example.npy'

    def __init__(self, root_folder=None, temporal_cov=None, spatial_cov=None, noise_scale=1.0):
        if root_folder is not None:
            self.root_folder = root_folder
        if temporal_cov is None:
            temporal_cov = np.load(os.path.join(self.root_folder, self.temporal_name))
        if spatial_cov is None:
            spatial_cov = np.load(os.path.join(self.root_folder + self.spatial_name))
        self.temporal_cov = temporal_cov
        self.spatial_cov = spatial_cov
        # self.noise_scale = np.float64(noise_scale)
        self.noise_scale = np.float32(noise_scale)
        

    def __call__(self, sample):
        if len(sample) == 2:
            wf, chan_nums = sample
        else:
            wf = sample
        
        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]
        w = wf.shape[1]

        assert self.temporal_cov.shape[0] == w

        n_neigh, _ = self.spatial_cov.shape
        waveform_length, _ = self.temporal_cov.shape

        noise = np.random.normal(size=(waveform_length, n_neigh))

        noise = np.matmul(noise.T, self.temporal_cov).T
        reshaped_noise = np.reshape(noise, (-1, n_neigh))

        the_noise = np.reshape(np.matmul(reshaped_noise, self.spatial_cov),
                        (waveform_length, n_neigh))
        
        # noise_start = np.random.choice(n_neigh - n_chans)
        if type(chan_nums) != np.int64: 
            chan_nums[chan_nums > n_neigh-1] = n_neigh - 1
            chan_nums[chan_nums < 0] = 0

#         noise_sel = np.random.choice(n_neigh, n_chans, replace=False)
        noise_wfs = self.noise_scale * the_noise[:, chan_nums].T
        wf = wf + noise_wfs

        return [wf, chan_nums]


class Collide(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    root_folder = '/home/jovyan/nyu47-templates/'
    temp_name = 'spikes_train.npy'

    def __init__(self, root_folder=None, templates=None, multi_chan=False):
        temp_name = self.temp_name
        if root_folder is not None:
            self.root_folder = root_folder
        if multi_chan:
            temp_name = 'multichan_' + temp_name
        if templates is None:
            templates = np.load(os.path.join(self.root_folder, temp_name))
        # assert isinstance(templates, (array, array))
        self.templates = templates

    def __call__(self, sample):
        if len(sample) == 2:
            wf, chan_nums = sample
        else:
            wf = sample
        
        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]
        w = wf.shape[1]
        
        temp_idx = np.random.randint(0, len(self.templates))
        temp_sel = self.templates[temp_idx]
        temp_sel = np.expand_dims(temp_sel, axis=0) if len(temp_sel.shape) == 1 else temp_sel
        
        scale = np.random.uniform(0.2, 1)
        shift = (2* np.random.binomial(1, 0.5)-1) * np.random.randint(5, 60)

        temp_sel = temp_sel * scale
        temp_sel = self.shift_chans(temp_sel, shift)

        wf = np.add(wf, temp_sel)

        return [wf, chan_nums]
    
    def shift_chans(self, wf, shift_):
        # use template feat_channel shifts to interpolate shift of all spikes on all other chans
        int_shift = int(math.ceil(shift_)) if shift_ >= 0 else -int(math.floor(shift_))
        curr_wf_pos = np.pad(wf, ((0, 0), (0, int_shift)), 'constant') 
        curr_wf_neg = np.pad(wf, ((0, 0), (int_shift, 0)), 'constant')
        if int(shift_)==shift_:
            ceil = int(shift_)
            temp = np.roll(curr_wf_pos,ceil,axis=1)[:, :-int_shift] if shift_ > 0 else np.roll(curr_wf_neg,ceil,axis=1)[:, int_shift:]
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            if shift_ > 0:
                temp = (np.roll(curr_wf_pos,ceil,axis=1)*(shift_-floor))[:, :-ceil] + (np.roll(curr_wf_pos,floor, axis=1)*(ceil-shift_))[:, :-ceil]
            else:
                temp = (np.roll(curr_wf_neg,ceil,axis=1)*(shift_-floor))[:, -floor:] + (np.roll(curr_wf_neg,floor, axis=1)*(ceil-shift_))[:, -floor:]
        wf_final = temp

        return wf_final

    
class Jitter(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, templates=None, up_factor=8, sample_rate=20000, shift=2):
        assert isinstance(up_factor, (int))
        assert isinstance(sample_rate, (int))
        self.templates = templates
        self.up_factor = up_factor
        self.sample_rate = sample_rate
        self.shift = shift

    def __call__(self, sample):
        if len(sample) == 2:
            wf, chan_nums = sample
        else:
            wf = sample

        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]
        w = wf.shape[1]
        
        up_temp = sp.signal.resample(
            x=wf,
            num=w*self.up_factor,
            axis=1)
            
        idx = (np.arange(0, w)[:,None]*self.up_factor + np.arange(self.up_factor))
        up_shifted_temp = np.transpose(up_temp[:, idx], (0, 2, 1))

        # temp_idx = np.random.choice(0, len(self.templates))
        # temp_sel = self.templates[temp_idx]

        # idx = np.arange(0, self.up_factor*w).reshape(-1, self.up_factor)
        # up_shifted_wfs = wf_upsamp[idx]
        # up_shifted_temps.unsqueeze(1)
        # up_shifted_temps = torch.cat(
        #     (up_shifted_temps,temp_sel),
        #     axis=1)
        

        shift = (2* np.random.binomial(1, 0.5)-1) * np.random.uniform(0, self.shift)
        
        idx_selection = np.random.choice(self.up_factor)
        idxs = np.array([idx_selection for i in range(n_chans)])
        wf = up_shifted_temp[np.arange(n_chans), idxs]
        wf = self.shift_chans(wf, shift)

        return [wf, chan_nums]

    def shift_chans(self, wf, shift_):
        # use template feat_channel shifts to interpolate shift of all spikes on all other chans
        int_shift = int(math.ceil(shift_)) if shift_ >= 0 else -int(math.floor(shift_))
        curr_wf_pos = np.pad(wf, ((0, 0), (0, int_shift)), 'constant') 
        curr_wf_neg = np.pad(wf, ((0, 0), (int_shift, 0)), 'constant')
        if int(shift_)==shift_:
            ceil = int(shift_)
            temp = np.roll(curr_wf_pos,ceil,axis=1)[:, :-int_shift] if shift_ > 0 else np.roll(curr_wf_neg,ceil,axis=1)[:, int_shift:]
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            if shift_ > 0:
                temp = (np.roll(curr_wf_pos,ceil,axis=1)*(shift_-floor))[:, :-ceil] + (np.roll(curr_wf_pos,floor, axis=1)*(ceil-shift_))[:, :-ceil]
            else:
                temp = (np.roll(curr_wf_neg,ceil,axis=1)*(shift_-floor))[:, -floor:] + (np.roll(curr_wf_neg,floor, axis=1)*(ceil-shift_))[:, -floor:]
        wf_final = temp

        return wf_final
    

class Crop(object):
    def __init__(self, prob=0.5, num_extra_chans=2):
        self.prob = prob
        self.num_extra_chans = num_extra_chans
        
    def __call__(self, sample):
        if len(sample) == 2:
            wf, chan_nums = sample
        else:
            wf = sample
        
        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        n_chans = wf.shape[0]
        n_times = wf.shape[1]
        
        max_chan_ind = math.floor(n_chans/2)

        apply = np.random.binomial(1, self.prob)
        if apply:
            shift = np.random.randint(-self.num_extra_chans, self.num_extra_chans+1)
            max_chan_ind += shift
        wf = wf[max_chan_ind-self.num_extra_chans:max_chan_ind+self.num_extra_chans+1]
        if type(chan_nums) != np.int64: 
            chan_nums = chan_nums[max_chan_ind-self.num_extra_chans:max_chan_ind+self.num_extra_chans+1]
        # in single channel case the wf will become 1 dimensional
        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)

        return [wf, chan_nums]


class PCA_Reproj(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    root_folder = '/datastores/dy016'
    spikes_file = 'spikes_train.npy'

    def __init__(self, root_folder=None, spikes_file=None, pca_dim=5):
        assert isinstance(pca_dim, (int))
        if root_folder is not None:
            self.root_folder = root_folder
        if spikes_file is not None:
            self.spikes_file = spikes_file
        self.spikes = np.load(os.path.join(self.root_folder, self.spikes_file))
        self.pca_dim = pca_dim
        self.pca_ = PCA(n_components=self.pca_dim).fit(self.spikes)
        self.spikes_mean = np.mean(self.spikes, axis=0)

    def __call__(self, sample):
        transform = self.pca_.transform(sample.reshape(1, -1))
        recon = self.pca_.inverse_transform(transform)[0]

        return recon

    
class ToWfTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample) == 2:
            wf, chan_nums = sample
        else:
            wf = sample
            
        if len(wf.shape) == 1:
            wf = np.expand_dims(wf, axis=0)
        
        return torch.from_numpy(wf.astype('float16'))


###### TORCH VERSIONS ######

# class AmpJitter(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """

#     def __call__(self, sample):
#         wf = sample
#         w = wf.shape[0]

#         uni = Uniform(torch.tensor([0.9]), torch.tensor([1.1]))
#         amp_jit = uni.rsample(torch.Size(w))

#         wf = torch.Mul(wf, amp_jit)
    
#         return wf

# class Noise(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """

#     def __call__(self, sample):
#         wf = sample
#         w = wf.shape[0]

#         mvn = MultivariateNormal(torch.zeros(w), torch.eye(w))
#         noise_wf = mvn.sample()
#         wf = wf + noise_wf

#         return wf

# class SmartNoise(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """
#     root_folder = '/Users/ankit/Documents/PaninskiLab/spike-psvae/notebook'
#     cov_name = 'temporal_cov_example.npy'

#     def __init__(self, temporal_cov=None):
#         if temporal_cov is None:
#             temporal_cov = np.load(self.root_folder + self.cov_name)
#         # assert isinstance(temporal_cov, (float, array))
#         self.temporal_cov = temporal_cov

#     def __call__(self, sample):
#         # wf = sample['wf']
#         wf = sample
#         w = wf.shape[0]

#         assert self.temporal_cov.shape[0] == w

#         mvn = MultivariateNormal(torch.zeros(w), self.temporal_cov)
#         noise_wf = mvn.sample()
#         wf = wf + noise_wf

#         return wf

# class Collide(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """
#     root_folder = '/Users/ankit/Documents/PaninskiLab/nyu47_templates/'
#     temp_name = 'kilo_hptp_temps.npy'

#     def __init__(self, templates=None):
#         if templates is None:
#             templates = np.load(self.root_folder+self.temp_name)
#         # assert isinstance(templates, (array, array))
#         self.templates = templates

#     def __call__(self, sample):
#         wf = sample['wf']
#         w = wf.shape[0]

#         temp_idx = torch.randint(0, len(self.templates), (1,))
#         temp_sel = torch.from_numpy(self.templates[temp_idx])

#         scale = Uniform(torch.tensor([0.2]), torch.tensor([1.0])).sample()
#         offset = torch.randint(5, 60, (1,))
#         shift = (torch.multiply(torch.tensor([2.0]), Bernoulli(torch.tensor([0.5])).sample()) - torch.tensor([1.0])).multiply(offset)

#         temp_sel = temp_sel.multiply(scale)
#         temp_sel = self.shift_chans(temp_sel, shift)

#         wf = wf.add(temp_sel)

#         return {'wf': wf}

#     def shift_chans(wf, shift):
#         # use template feat_channel shifts to interpolate shift of all spikes on all other chans
#         int_shift = torch.ceil(shift) if shift.item() >= 0 else torch.floor(shift).multiply(-1)
#         int_shift = int(int_shift.view())
#         curr_wf_pos = F.pad(wf, (0, int_shift), 'constant') 
#         curr_wf_neg = F.pad(wf, (int_shift, 0), 'constant')

#         if shift > 0:
#             wf_final = torch.roll(curr_wf_pos,int_shift,axis=0)[:-int_shift]
#         else:
#             wf_final = torch.roll(curr_wf_neg,int_shift,axis=0)[int_shift:]
        
#         return wf_final

# class Jitter(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """

#     def __init__(self, templates=None, up_factor=8, sample_rate=20000):
#         # assert isinstance(templates, (array, array))
#         assert isinstance(up_factor, (int))
#         assert isinstance(sample_rate, (int))
#         self.templates = templates
#         self.up_factor = up_factor
#         self.sample_rate = sample_rate

#     def __call__(self, sample):
#         wf = sample['wf']
#         w = wf.shape[0]

#         resample = Resample(self.sample_rate, self.up_factor * self.sample_rate)
#         wf_upsamp = resample(wf).t()

#         # temp_idx = torch.randint(0, len(self.templates), (1,))
#         # temp_sel = self.templates[temp_idx]

#         idx = torch.arange(0, self.up_factor*w).reshape(-1, self.up_factor)
#         up_shifted_wfs = wf_upsamp[idx]
#         # up_shifted_temps.unsqueeze(1)
#         # up_shifted_temps = torch.cat(
#         #     (up_shifted_temps,temp_sel),
#         #     axis=1)
        
#         offset = torch.randint(0, 3, (1,))
#         shift = (torch.multiply(torch.tensor([2.0]), Bernoulli(torch.tensor([0.5])).sample()) - torch.tensor([1.0])).multiply(offset)

#         idx_selection = torch.randint(self.up_factor)
#         wf = up_shifted_wfs[idx_selection]
#         wf = self.shift_chans(wf, shift)

#         return wf

#     def shift_chans(wf, shift):
#         # use template feat_channel shifts to interpolate shift of all spikes on all other chans
#         int_shift = torch.ceil(shift) if shift.item() >= 0 else torch.floor(shift).multiply(-1)
#         int_shift = int(int_shift.view())
#         curr_wf_pos = F.pad(wf, (0, int_shift), 'constant') 
#         curr_wf_neg = F.pad(wf, (int_shift, 0), 'constant')

#         if shift > 0:
#             wf_final = torch.roll(curr_wf_pos,int_shift,axis=0)[:-int_shift]
#         else:
#             wf_final = torch.roll(curr_wf_neg,int_shift,axis=0)[int_shift:]
        
#         return wf_final
    
#     # def shift_chans(wfs, shifts):
#     #     wfs_final= torch.zeros(wfs.size(), dtype=torch.float32)

#     #     for k, shift in enumerate(shifts):
#     #         wf = wf[k]
#     #         # use template feat_channel shifts to interpolate shift of all spikes on all other chans
#     #         int_shift = torch.ceil(shift) if shift.item() >= 0 else torch.floor(shift).multiply(-1)
#     #         int_shift = int(int_shift.view())
#     #         curr_wf_pos = F.pad(wf, (0, int_shift), 'constant') 
#     #         curr_wf_neg = F.pad(wf, (int_shift, 0), 'constant')

#     #         if shift > 0:
#     #             wfs_final[k] = torch.roll(curr_wf_pos,int_shift,axis=0)[:-int_shift]
#     #         else:
#     #             wfs_final[k] = torch.roll(curr_wf_neg,int_shift,axis=0)[int_shift:]
        
#     #     return wfs_final