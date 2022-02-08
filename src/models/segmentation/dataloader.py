import os
import pytorch_lightning as pl

from typing import Optional
from torch.utils.data import DataLoader

import transformers
from datasets import SolarPanelsDataset


class SolarPanelsDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, classes, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.get_dataset('train')
        self.val_dataset = self.get_dataset('val')
        self.test_dataset = self.get_dataset('test')

    def get_dataset(self, phase):
        return SolarPanelsDataset(
            images_dir=os.path.join(self.data_dir, f'{phase}/images'),
            masks_dir=os.path.join(self.data_dir, f'{phase}/masks'),
            classes=self.classes,
            augmentation=self.get_augmentation(phase),
            preprocessing=transformers.get_preprocessing(),
        )

    @staticmethod
    def get_augmentation(phase):
        if phase == 'train':
            return transformers.get_training_augmentation()
        else:
            return transformers.get_validation_augmentation()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,
                          shuffle=False, num_workers=self.num_workers)
