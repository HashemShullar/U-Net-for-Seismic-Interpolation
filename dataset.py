import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class Seismic(Dataset):
    def __init__(self, image_dir, target_dir, transform=None, validation_flag = 1):
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.validation_flag = validation_flag

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        target_path = os.path.join(self.target_dir, self.images[index])
        target = np.loadtxt(target_path, delimiter=',')



        if self.validation_flag == 1:
            image = np.loadtxt(img_path, delimiter=',')



        else:
            image   = np.loadtxt(target_path, delimiter=',')

            # Automated trace removal:

            rr = np.arange(0, 79)
            factor_range = np.arange(10, 71, 10)
            factor = np.random.choice(factor_range, size=(1,), replace=True, p=np.array([0.1, 0.1, 0.15, 0.15, 0.3, 0.15, 0.05]))[0] 
            rando  = np.random.choice(rr, size=(1, factor), replace=False)

            image[:, rando] = 0


        if self.transform is not None:

            augmentations = self.transform(image=image, mask=target)
            image = augmentations["image"]
            target = augmentations["mask"]


        return image, target
