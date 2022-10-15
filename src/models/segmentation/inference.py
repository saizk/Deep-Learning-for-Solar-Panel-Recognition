import time

from PIL import Image

import transformers
import postprocess as postp
from model import SolarPanelsModel
from dataloader import *
from utils import *


def post_process(image):
    image = postp.binary_closing(image)
    image = postp.binary_fill_holes(image)
    image = postp.erosion(image)
    image = postp.dilation(image)
    return image


def inference(model_path, params, save=False):

    model = load_model(SolarPanelsModel, model_path, params)

    gm_module = GoogleMapsDataModule(
        data_dir=params['data_dir'],
        # train_augmentation=params['train_augmentation'],
        valid_augmentation=params['valid_augmentation'],
        preprocessing=params['preprocessing'],
        num_workers=params['num_workers'],
    )

    masks = []
    model.eval()
    with torch.no_grad():
        for idx, image in enumerate(gm_module.inference_dataloader()):
            pr_mask = model.model.predict(image).cpu()
            pr_mask = (pr_mask.squeeze().numpy().round())
            post_mask = post_process(pr_mask)
            if save:
                mask = overlap(image, mask)
                mask_img = Image.fromarray(mask)
                mask_img.save(f'{params["data_dir"]}/plots{idx}.png')
                visualize(image=image, mask=mask)
                time.sleep(5)

            masks.append(post_mask)

    return masks


if __name__ == '__main__':

    data_dir = '../../../data/gmaps/images'
    results_dir = '../../../models'
    model_name = f'{results_dir}/unetplusplus_se_resnext101_32x4d_e5.pth'

    model_params = {
        'data_dir': data_dir,
        'results_dir': results_dir,

        'train_augmentation': transformers.get_training_augmentation,
        'valid_augmentation': transformers.get_validation_augmentation,
        'preprocessing': transformers.get_preprocessing,

        'architecture': 'UnetPlusPlus',
        'encoder': 'se_resnext101_32x4d',
        'classes': ['solar_panel'],
        'num_workers': 4
    }
    inference(model_name, model_params)
