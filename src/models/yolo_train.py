import torch.cuda
import yolo.yolov5.train as yolo_train


def train_yolo(**kwargs):
    params = get_yolo_params(**kwargs)
    yolo_train.run(**params)


def get_yolo_params(name, epochs=50, project='models', weights='yolov5s',
                    evolve=0, device='cuda:0'):
    yolo_parameters = {
        'data': 'yolo/sp_dataset.yaml',
        # 'weights': f'{project}/{weights}/weights/best.pt',
        'weights': weights + '.pt' if not weights.endswith('.pt') else '',
        'imgsz': 256,
        'batch_size': 16,
        'workers': 4,
        'project': project,
        'name': name,
        'epochs': epochs,
        'device': device
    }
    if evolve:
        yolo_parameters['evolve'] = evolve
        yolo_parameters['name'] += '_hyp'
    return yolo_parameters


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    epochs = 10
    model = 'yolov5s'
    model_name = f'{model}_{epochs}_test'

    train_yolo(
        name=model_name,
        epochs=epochs,
        project='models',  # evolve=300,
        weights=model, device=device
    )


if __name__ == '__main__':
    main()
