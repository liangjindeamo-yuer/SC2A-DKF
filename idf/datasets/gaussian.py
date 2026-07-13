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
from idf.utils.noise import add_Gaussian_noise
from typing import Tuple

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

class GaussianDataset(Dataset) :
    def __init__(self, dataroot:str, patch_size:int, augmentation:bool,
                 noise_level:Tuple[int, int], channel_wise_noise:bool,
                 preload:bool, parallel_preload:bool, test:bool) :
        super(GaussianDataset, self).__init__()
        
        # Initialize Variables
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.test = test
        self.preload = preload
        self.augmentation = augmentation

        self.noise_level = noise_level
        self.channel_wise_noise = channel_wise_noise
        
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

        clean_patch = clean_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        noisy_patch, noise_level = add_Gaussian_noise(clean_patch, 
                                                      noise_level1=self.noise_level[0], 
                                                      noise_level2=self.noise_level[1],
                                                      channel_wise=self.channel_wise_noise)

        img_item = {}
        img_item['GT'] = clean_patch
        img_item['LQ'] = noisy_patch
        img_item['file_name'] = self.cleanDataset[0][index]
        img_item['noise_level'] = noise_level
        
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
    
  