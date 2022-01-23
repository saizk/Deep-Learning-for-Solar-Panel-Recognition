import os
from pathlib import Path

import segmentation_models_pytorch as smp
from dataset import SolarPanelsDataset
from transformers import *


def get_dataset(phase, augmentation, params):
    x_dir = os.path.join(params['data_dir'], f'{phase}/images')
    y_dir = os.path.join(params['data_dir'], f'{phase}/masks')

    return SolarPanelsDataset(
        x_dir, y_dir,
        classes=params['classes'],
        augmentation=augmentation(),
        preprocessing=get_preprocessing(smp.encoders.get_preprocessing_fn(params['encoder'])),
    )


def get_model(model, encoder, n_classes, activation):
    return model(encoder, classes=n_classes, activation=activation)


def get_optimizer(model, optimizer, lr):
    return optimizer(params=model.parameters(), lr=lr)


def get_model_info(model_name):
    raw_name = '_'.join(model_name.split('/')[-1].split('.')[0].split('_')[:-1])
    epochs = int(model_name.split('.')[0].split('_')[-1])
    return raw_name, epochs


def model_exists(model_name):
    parent = Path(model_name).parent
    name, _ = get_model_info(model_name)
    for model in os.listdir(parent):
        if model.startswith(name):
            return os.path.join(parent, model)