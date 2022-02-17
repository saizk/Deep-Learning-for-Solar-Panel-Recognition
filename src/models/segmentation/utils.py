import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_model(model_cls, model_path, params):
    model = model_cls(
        arch=params['architecture'],
        encoder=params['encoder'],
        in_channels=3,
        out_classes=len(params['classes']),
        model_params=params
    )
    model.load_state_dict(torch.load(model_path), strict=False)
    return model


def get_model_info(model_name):
    raw_name = '_'.join(model_name.split('/')[-1].split('.')[0].split('_')[:-1])
    epochs = int(model_name.split('.')[0].split('_')[-1])
    return raw_name, epochs


def visualize(**images):
    """
    Helper function for data visualization
    Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def overlap(image, mask):
    color = np.array([255, 0, 0], dtype='uint8')  # color to fill
    masked_img = np.where(mask[..., None], color, image)
    out = cv2.addWeighted(image, 0.7, masked_img, 0.2, 0)
    return out
