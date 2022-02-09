import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, images_dir):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

    def __len__(self):
        return len(self.ids)


class SolarPanelsDataset(BaseDataset):
    """Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        classes (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """

    CLASSES = ['solar_panel']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        super().__init__(images_dir)
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0] + '_label.png') for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image, mask = cv2.imread(self.images_fps[i]), cv2.imread(self.masks_fps[i], 0)  # read data

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

        # extract certain classes from mask (e.g. cars)
        masks = [(mask != v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        if self.augmentation:  # apply augmentations
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:  # apply preprocessing
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask


class GoogleMapsDataset(BaseDataset):
    """Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """

    def __init__(self, images_dir, augmentation=None, preprocessing=None):
        super().__init__(images_dir)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])  # read data
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation:  # apply augmentations
            sample = self.augmentation(image=image)
            image = sample['image']

        if self.preprocessing:  # apply preprocessing
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image
