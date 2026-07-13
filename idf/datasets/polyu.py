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

class PolyUDataset(Dataset) :
    def __init__(self, dataroot:str, patch_size:int, augmentation:bool,
                 preload:bool, parallel_preload:bool, test:bool) :
        super(PolyUDataset, self).__init__()
        
        # Initialize Variables
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.test = test
        self.preload = preload
        self.augmentation = augmentation
        
        # Get Dataset Instances
        self.noisyDataset, self.cleanDataset = self.getPathList()    

        self.GT_dir = [join(self.cleanDataset[1], fn) for fn in self.cleanDataset[0]]
        self.LQ_dir = [join(self.noisyDataset[1], fn) for fn in self.noisyDataset[0]]

        if self.preload:
            if parallel_preload:
                # Preload images into RAM in parallel
                with ThreadPoolExecutor() as executor:
                    self.GT = list(executor.map(self.load_image, self.GT_dir))
                with ThreadPoolExecutor() as executor:
                    self.LQ = list(executor.map(self.load_image, self.LQ_dir))
            else:
                self.GT, self.LQ = [], []
                for dir in self.GT_dir:
                    self.GT.append(np.array(Image.open(dir).convert('RGB')))
                for dir in self.LQ_dir:
                    self.LQ.append(np.array(Image.open(dir).convert('RGB')))
    
    def load_image(self, img_path):
        image = np.array(Image.open(img_path).convert('RGB'))
        return image

    def __getitem__(self, index) :
        # Load Data
        if self.preload:
            noisy = self.LQ[index]
            clean = self.GT[index]
        else:
            noisy = np.array(Image.open(self.LQ_dir[index]).convert("RGB"))
            clean = np.array(Image.open(self.GT_dir[index]).convert("RGB"))

        # Get Label
        sensor, iso = self.noisyDataset[0][index].split("_")[0], self.noisyDataset[0][index].split("_")[3]
        label = f"{sensor}_{iso}"

        if not self.test and self.augmentation:
            h, w, _ = clean.shape
            rnd_h = random.randint(0, max(0, h - self.patch_size))
            rnd_w = random.randint(0, max(0, w - self.patch_size))
            clean_patch = clean[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            noisy_patch = noisy[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            mode = random.randint(0, 7)
            clean_patch = augment_img(clean_patch, mode)
            noisy_patch = augment_img(noisy_patch, mode)
        else:
            clean_patch = clean
            noisy_patch = noisy

        clean_patch = clean_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        noisy_patch = noisy_patch.transpose(2, 0, 1).astype(np.float32) / 255.

        img_item = {}
        img_item['GT'] = clean_patch
        img_item['LQ'] = noisy_patch
        img_item['file_name'] = self.noisyDataset[0][index]
        img_item['metadata'] = label
        
        return img_item

    def __len__(self):
        return len(self.noisyDataset[0])

    def getPathList(self) :            
        noisyPath = join(self.dataroot, "noisy_256")
        cleanPath = join(self.dataroot, "clean_256")
    
        # Create List Instance for Adding Dataset Path
        noisyPathList = listdir(noisyPath)
        cleanPathList = listdir(cleanPath)
        
        # Create List Instance for Adding File Name
        noisyNameList = [imageName for imageName in noisyPathList if ".jpg" in imageName or ".JPG" in imageName]
        cleanNameList = [imageName for imageName in cleanPathList if ".jpg" in imageName or ".JPG" in imageName]
        
        # Sort List Instance
        noisyNameList = natsorted(noisyNameList)
        cleanNameList = natsorted(cleanNameList)
        
        return (noisyNameList, noisyPath), (cleanNameList, cleanPath)