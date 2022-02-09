import os
from pathlib import Path


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
