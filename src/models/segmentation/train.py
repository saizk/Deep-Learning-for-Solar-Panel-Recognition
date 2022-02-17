import torch
import segmentation_models_pytorch as smp

import transformers
from model import SolarPanelsModel
from dataloader import *


def train(params, device):
    sp_module = SolarPanelsDataModule(
        data_dir=params['data_dir'],
        classes=params['classes'],
        train_augmentation=params['train_augmentation'],
        valid_augmentation=params['valid_augmentation'],
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
    )
    model = SolarPanelsModel(
        arch=params['architecture'],
        encoder=params['encoder'],
        in_channels=3,
        out_classes=len(params['classes']),
        model_params=params
    )
    trainer = pl.Trainer(gpus=-1, max_epochs=params['epochs'])
    trainer.fit(model, datamodule=sp_module)

    valid_metrics = trainer.validate(model, datamodule=sp_module, verbose=False)

    if not os.path.exists(params['model_name']):
        torch.save(trainer.model.state_dict(), params['model_name'])

    return trainer, valid_metrics


if __name__ == '__main__':
    arch = 'UnetPlusPlus'
    encoder = 'se_resnext101_32x4d'
    epochs = 50

    data_dir = '../../../data'
    results_dir = '../../../models'

    model_params = {
        'data_dir': data_dir,
        'results_dir': results_dir,
        'model_name': f'{results_dir}/{arch.lower()}_{encoder}_e{epochs}.pth',

        'architecture': arch,
        'encoder': encoder,
        'classes': ['solar_panel'],
        'lr': 0.0001,
        'epochs': epochs,
        'batch_size': 16,
        'num_workers': 4,

        'train_augmentation': transformers.get_training_augmentation,
        'valid_augmentation': transformers.get_validation_augmentation,
        'preprocessing': transformers.get_preprocessing,

        'loss': smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True),
        'optimizer': torch.optim.Adam
    }
    trainer, metrics = train(model_params, device='cuda')
    print(metrics)
