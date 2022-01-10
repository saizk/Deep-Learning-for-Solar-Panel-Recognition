import torch
from torch.utils.data import DataLoader

from segmentation.utils import *
from segmentation.transformers import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def gen_params(arch, encoder, classes,
               data_dir, lr, batch_size, epochs):
    return {
        'architecture': arch,
        'encoder': encoder,
        'model_name': f'models_pytorch/{arch.__name__.lower()}_{encoder}_model_{epochs}.pth',
        'data_dir': data_dir,

        'classes': classes,
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size,

        'loss': smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True),
        'metrics': [smp.metrics.IoU(threshold=0.5),
                    smp.metrics.Fscore(threshold=0.5)],
        'optimizer': torch.optim.Adam
    }


def train(train_params, device, verbose=True):
    model_name = model_exists(train_params['model_name'])
    n_classes = 1 if len(train_params['classes']) == 1 else (
            len(train_params['classes']) + 1)  # case for binary and multiclass segmentation

    if model_name is not None:

        model = torch.load(model_name)
        raw_name, prev_epochs = get_model_info(model_name)

        if prev_epochs == 0:
            print(f'There already exists a model: {model_name}')
            return

        train_params['epochs'] -= prev_epochs

    else:
        model = get_model(
            model=train_params['architecture'],
            encoder=train_params['encoder'],
            activation='sigmoid' if n_classes == 1 else 'softmax',
            n_classes=n_classes,
        )

    train_dataset = get_dataset('train', get_training_augmentation, train_params)  # Dataset for training images
    valid_dataset = get_dataset('val', get_validation_augmentation, train_params)  # Dataset for validation images

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        num_workers=2
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    return trainloop(model, train_loader, valid_loader, train_params, device, verbose)


def trainloop(model, train_loader, valid_loader, train_params, device, verbose):
    optimizer = get_optimizer(model, train_params['optimizer'], train_params['lr'])

    train_epoch = smp.train.TrainEpoch(
        model,
        loss=train_params['loss'],
        metrics=train_params['metrics'],
        optimizer=optimizer,
        device=device,
        verbose=verbose,
    )
    valid_epoch = smp.train.ValidEpoch(
        model,
        loss=train_params['loss'],
        metrics=train_params['metrics'],
        device=device,
        verbose=verbose,
    )

    max_score = 0
    print(train_params['model_name'])

    for epoch in range(train_params['epochs']):
        print(f'\nEpoch: {epoch + 1}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

    if not os.path.exists(train_params['model_name']):
        torch.save(model, train_params['model_name'])


def main():
    ARCHITECTURE = smp.FPN
    ENCODER = 'efficientnet-b3'
    CLASSES = ['solar_panel']
    BATCH_SIZE = 16
    LR = 0.0001
    EPOCHS = 25

    DATA_DIR = './data/'
    DEVICE = 'cuda'

    train_params = gen_params(
        ARCHITECTURE, ENCODER, CLASSES,
        DATA_DIR, LR, BATCH_SIZE, EPOCHS
    )
    train(train_params, DEVICE)


if __name__ == '__main__':
    main()
