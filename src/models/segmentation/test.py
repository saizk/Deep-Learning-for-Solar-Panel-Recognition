import transformers
from dataloader import *
from utils import *
from model import SolarPanelsModel


def test(model_path, params):
    model = load_model(SolarPanelsModel, model_path, params)
    datamodule = SolarPanelsDataModule(
        data_dir=params['data_dir'],
        classes=params['classes'],
        train_augmentation=params['train_augmentation'],
        valid_augmentation=params['valid_augmentation'],
        preprocessing=params['preprocessing'],
    )
    trainer = pl.Trainer(gpus=-1)
    test_metrics = trainer.test(model, datamodule=datamodule, verbose=False)

    return trainer, test_metrics


if __name__ == '__main__':

    data_dir = '../../../data'
    results_dir = '../../../models'
    model_name = f'{results_dir}/unetplusplus_se_resnext101_32x4d_e5.pth'

    model_params = {
        'data_dir': data_dir,
        'results_dir': results_dir,
        'num_workers': 8,

        'train_augmentation': transformers.get_training_augmentation,
        'valid_augmentation': transformers.get_validation_augmentation,
        'preprocessing': transformers.get_preprocessing,

        'architecture': 'UnetPlusPlus',
        'encoder': 'se_resnext101_32x4d',
        'classes': ['solar_panel'],
    }
    _, metrics = test(model_name, model_params)
    print(metrics)
