import numpy as np
import random
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from os import listdir
from os.path import join
import numpy as np
from natsort import natsorted
import torch
# from idf.utils.noise import add_Gaussian_noise
from typing import Tuple
from idf.utils.degradation import add_gaussian_noise, add_poisson_noise
import omegaconf

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def _get_param_value(param_config, default_value, is_integer=False):
    """Helper function to get a parameter value.
    If param_config is a list/tuple [min, max], sample uniformly.
    Otherwise, use param_config if it's a number, or default_value.
    is_integer ensures integer sampling for params like kernel_size.
    """
    if isinstance(param_config, (list, tuple, omegaconf.ListConfig)) and len(param_config) == 2:
        min_val, max_val = param_config
        # Ensure min_val <= max_val for uniform/randint
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        
        if is_integer:
            if int(min_val) == int(max_val): return int(min_val)
            return random.randint(int(min_val), int(max_val))
        else:
            return random.uniform(min_val, max_val)
    elif isinstance(param_config, (int, float)):
        return param_config
    assert False, f"Invalid parameter config: {param_config}. Expected list/tuple of length 2 or a number."
    return default_value


class SyntheticDataset(Dataset) :
    def __init__(self, dataroot:str, patch_size:int, augmentation:bool,
                 preload:bool, parallel_preload:bool, test:bool,
                 noise_types:list=None, 
                 noise_params:dict=None,) :
        super(SyntheticDataset, self).__init__()

        # Initialize Variables
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.test = test
        self.preload = preload
        self.augmentation = augmentation

        self.noise_types = noise_types if noise_types else []
        self.noise_params = noise_params if noise_params else {}
        
        # Get Dataset Instances
        self.cleanDataset = self.getPathList()    

        self.GT_dir = [join(self.cleanDataset[1], fn) for fn in self.cleanDataset[0]]

        if self.preload:
            if parallel_preload:
                # Preload images into RAM in parallel
                with ThreadPoolExecutor() as executor:
                    self.GT = list(executor.map(self.load_image, self.GT_dir))
            else:
                self.GT= []
                for dir in self.GT_dir:
                    self.GT.append(np.array(Image.open(dir).convert('RGB')))
    
    def load_image(self, img_path):
        image = np.array(Image.open(img_path).convert('RGB'))
        return image

    def __getitem__(self, index) :
        # Load Data
        if self.preload:
            clean = self.GT[index]
        else:
            clean = np.array(Image.open(self.GT_dir[index]).convert("RGB"))

        h, w, _ = clean.shape
        rnd_h = random.randint(0, max(0, h - self.patch_size))
        rnd_w = random.randint(0, max(0, w - self.patch_size))
        clean_patch = clean[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            clean_patch = augment_img(clean_patch, mode)
        
        current_noisy_patch = clean_patch.astype(np.float32) / 255.

        metadata = {}
        
        if 'gaussian' in self.noise_types:
            params = self.noise_params.get('gaussian', {})
            sigma = _get_param_value(params.get('sigma'), 0.0)
            
            current_noisy_patch = add_gaussian_noise(current_noisy_patch,
                                                        sigma=sigma,
                                                        clip=True,
                                                        rounds=False,
                                                        gray_noise=False)
            metadata['gaussian'] = sigma
        if 'poisson' in self.noise_types:
            params = self.noise_params.get('poisson', {})
            alpha = _get_param_value(params.get('alpha'), 0.0)
            current_noisy_patch = add_poisson_noise(current_noisy_patch,
                                                    scale=alpha,
                                                    clip=True,
                                                    rounds=False,
                                                    gray_noise=False)
            metadata['poisson'] = alpha
        noisy_patch = current_noisy_patch.transpose(2, 0, 1)
        clean_patch = clean_patch.transpose(2, 0, 1).astype(np.float32) / 255.

        img_item = {}
        img_item['GT'] = clean_patch
        img_item['LQ'] = noisy_patch
        img_item['file_name'] = self.cleanDataset[0][index]
        img_item['metadata'] = metadata
        
        return img_item

    def __len__(self):
        return len(self.cleanDataset[0])

    def getPathList(self) :
        # Get Dataset Path
        cleanPath = self.dataroot
    
        # Create List Instance for Adding Dataset Path
        cleanPathList = listdir(cleanPath)
        
        # Create List Instance for Adding File Name
        cleanNameList = [imageName for imageName in cleanPathList if imageName.split('.')[-1] in ["png", "bmp", "jpg"]]
        
        # Sort List Instance
        cleanNameList = natsorted(cleanNameList)
        
        return (cleanNameList, cleanPath)
    
  