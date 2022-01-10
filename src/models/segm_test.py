import torch
from torch.utils.data import DataLoader
from segmentation.utils import *


def gen_test_params(data_dir, encoder):
    return {
        'encoder': encoder,
        'data_dir': data_dir,
        'classes': ['solar_panel'],

        'loss': smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True),
        'metrics': [smp.metrics.IoU(threshold=0.5),
                    smp.metrics.Fscore(threshold=0.5)],
    }


def test(model, test_params, device):

    test_dataset = get_dataset('test', get_validation_augmentation, test_params)  # Dataset for validation images
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    test_epoch = smp.train.ValidEpoch(
        model=model,
        loss=test_params['loss'],
        metrics=test_params['metrics'],
        device=device,
    )

    logs = test_epoch.run(test_dataloader)

    return test_dataset, logs


def main():
    MODEL_NAME = ''
    ENCODER = ''
    DATA_DIR = './data/'
    DEVICE = 'cuda'

    model = torch.load(MODEL_NAME)
    test_params = gen_test_params(DATA_DIR, ENCODER)
    test(model, test_params, DEVICE)


if __name__ == '__main__':
    main()
