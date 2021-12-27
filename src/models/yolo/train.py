import torch.cuda
import yolov5.train as yolo_train


def train_yolo(**kwargs):
    yolo_train.run(**kwargs)


def get_yolo_params(name, epochs=50, project='models', weights='yolov5l',
                    evolve=0, device='cuda:0'):
    yolo_parameters = {
        'data': 'sp_dataset.yaml',
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
    models = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
    epoch_range = [25, 50, 100]

    # for model in models:
    #     print('MODEL:', model.title())
    #     for epochs in [50]:
    #         print('EPOCHS:', epochs)
    #         params = get_yolo_params(epochs=epochs, project=models_path,
    #                                  model=model, device=device)
    #         train_yolo(**params)

    epochs = 10
    model = 'yolov5s'
    model_name = f'{model}_{epochs}_test'

    params = get_yolo_params(model_name, epochs=epochs,
                             project='models',  # evolve=300,
                             weights=model, device=device)
    train_yolo(**params)


if __name__ == '__main__':
    main()
