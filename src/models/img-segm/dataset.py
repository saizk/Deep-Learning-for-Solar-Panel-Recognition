import os
import cv2
import numpy as np
from abc import ABC
from torch.utils.data import Dataset
from typing import Optional




class SolarPanelsDataset(Dataset):

    rle_cols = [f"rle{i}" for i in range(N_CLASSES)]
    bin_cols = [f"c{i}" for i in range(N_CLASSES)]

    @property
    def img_folder(self):
        raise NotImplementedError

    def __init__(
            self,
            images_dir: str,
            masks_dir: str,
            fnames: Optional[list] = None,
            transforms: callable = None,
    ):
        super().__init__()
        self.fnames = fnames if fnames is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms

    def read_rgb(self, idx):
        f = self.fnames[idx]
        img = cv2.imread(str(self.data_dir / f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return f, img

    def rle(self, idx):
        return self.df.iloc[idx][self.rle_cols]

    def binary(self, idx):
        return self.df.iloc[idx][self.bin_cols]

    def __len__(self):
        return len(self.fnames)


class SolarPanelsTrain(SolarPanelsDataset):

    img_folder = "train_images"

    def __init__(self, df, data_dir, transforms):
        super().__init__(df, data_dir, transforms)
        self.transforms = transforms

    def __getitem__(self, idx):
        mask = make_mask(self.rle(idx))
        _, img = self.read_rgb(idx)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        mask = mask.permute(2, 0, 1)  # HxWxC => CxHxW
        return img, mask


class SolarPanelsTest(SolarPanelsDataset):

    img_folder = "test_images"

    def __init__(self, df, data_dir, transforms):
        super().__init__(df, data_dir, transforms)
        self.transforms = transforms

    def __getitem__(self, idx):
        f, img = self.read_rgb(idx)
        augmented = self.transforms(image=img)
        img = augmented["image"]
        return f, img
