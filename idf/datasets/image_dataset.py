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
from idf.datasets.utils import augment_img

class PairedDataset(Dataset) :
    def __init__(self, dataroot:str, patch_size:int, augmentation:bool,
                 preload:bool, parallel_preload:bool, test:bool,
                 lq_folder:str='noisy', gt_folder:str='clean') :
        super(PairedDataset, self).__init__()
        
        # Initialize Variables
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.test = test
        self.preload = preload
        self.augmentation = augmentation
        
        # Get Dataset Instances
        self.lq_folder, self.gt_folder = lq_folder, gt_folder
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

        if not self.test:
            h, w, _ = clean.shape
            rnd_h = random.randint(0, max(0, h - self.patch_size))
            rnd_w = random.randint(0, max(0, w - self.patch_size))
            clean_patch = clean[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            noisy_patch = noisy[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        else:
            if self.patch_size is not None:
                # perform center crop
                h, w, _ = clean.shape
                start_h = (h - self.patch_size) // 2
                start_w = (w - self.patch_size) // 2
                
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size

                clean_patch = clean[start_h:end_h, start_w:end_w, :]
                noisy_patch = noisy[start_h:end_h, start_w:end_w, :]
            else:
                clean_patch = clean
                noisy_patch = noisy
            
        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            clean_patch = augment_img(clean_patch, mode)
            noisy_patch = augment_img(noisy_patch, mode)

        clean_patch = clean_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        noisy_patch = noisy_patch.transpose(2, 0, 1).astype(np.float32) / 255.

        img_item = {}
        img_item['GT'] = clean_patch
        img_item['LQ'] = noisy_patch
        img_item['file_name'] = self.noisyDataset[0][index]
        # img_item['metadata'] = label
        
        return img_item

    def __len__(self):
        return len(self.noisyDataset[0])

    def getPathList(self) :            
        noisyPath = join(self.dataroot, self.lq_folder)
        cleanPath = join(self.dataroot, self.gt_folder)
    
        # Create List Instance for Adding Dataset Path
        noisyPathList = listdir(noisyPath)
        cleanPathList = listdir(cleanPath)
        
        # Create List Instance for Adding File Name
        noisyNameList = [imageName for imageName in noisyPathList if imageName.split(".")[-1] in ["png", "tif"]]
        cleanNameList = [imageName for imageName in cleanPathList if imageName.split(".")[-1] in ["png", "tif"]]
        
        # Sort List Instance
        noisyNameList = natsorted(noisyNameList)
        cleanNameList = natsorted(cleanNameList)
        
        return (noisyNameList, noisyPath), (cleanNameList, cleanPath)
    