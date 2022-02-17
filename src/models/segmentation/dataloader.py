import os
import pytorch_lightning as pl

from typing import Optional
from torch.utils.data import DataLoader

from datasets import SolarPanelsDataset, GoogleMapsDataset
from transformers import *


class BaseDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=1, num_workers=2,
                 train_augmentation=None, valid_augmentation=None, preprocessing=None):
        super().__init__()
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_augmentation = train_augmentation
        self.valid_augmentation = valid_augmentation
        self.preprocessing = preprocessing

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.get_dataset('train')
        self.val_dataset = self.get_dataset('val')
        self.test_dataset = self.get_dataset('test')

    def get_dataset(self, phase):
        pass

    def get_augmentation(self, phase):
        if phase == 'train':
            return self.train_augmentation()
        else:
            return self.valid_augmentation()


class SolarPanelsDataModule(BaseDataModule):

    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.classes = classes

    def get_dataset(self, phase):
        return SolarPanelsDataset(
            images_dir=os.path.join(self.data_dir, f'{phase}/images'),
            masks_dir=os.path.join(self.data_dir, f'{phase}/masks'),
            classes=self.classes,
            augmentation=self.get_augmentation(phase),
            preprocessing=self.preprocessing()
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,
                          shuffle=False, num_workers=self.num_workers)


class GoogleMapsDataModule(BaseDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inf_dataloader = self.get_dataset('val')

    def get_dataset(self, phase):
        return GoogleMapsDataset(
            images_dir=self.data_dir,
            augmentation=self.get_augmentation(phase),
            preprocessing=self.preprocessing(),
        )

    def inference_dataloader(self):
        return DataLoader(self.inf_dataloader, batch_size=1,
                          shuffle=False, num_workers=self.num_workers)
