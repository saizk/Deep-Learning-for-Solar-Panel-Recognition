import torch.cuda

import yolov5.train as yolo_train


def train_yolo(**kwargs):
    yolo_train.run(**kwargs)


def get_yolo_params(epochs=50, model='yolov5l', device='cuda:0'):
    return {
        'data': 'sp_dataset.yaml',
        'weights': f'{model}.pt',
        'imgsz': 256,
        'batch_size': 16,
        'workers': 4,
        'project': 'models',
        'name': f'{model}_{epochs}',
        'epochs': epochs,
        'device': device
    }


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    models = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']

    for model in models:
        print('MODEL:', model.title())
        for epochs in [25, 50, 100]:
            print('EPOCHS:', epochs)
            params = get_yolo_params(epochs=epochs, model=model)
            train_yolo(**params)



if __name__ == '__main__':
    main()

# py train.py --img 256 --batch 32 --epochs 50 --data sp_dataset.yaml --weights yolov5m.pt (To train)