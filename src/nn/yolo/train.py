import yolov5.train as yolo_train


def train_yolo(**kwargs):
    yolo_train.run(**kwargs)


def get_yolo_params(epochs=50, model='yolov5l'):
    return {
        'data': 'sp_dataset.yaml',
        'weights': f'{model}.pt',
        'imgsz': 256,
        'batch_size': 16,
        'workers': 4,
        'project': 'models',
        'name': f'{model}_{epochs}',
        'epochs': epochs,
    }


def main(net='yolo'):

    if net == 'yolo':
        models = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
        for model in models:
            print('MODEL:', model.title())
            for epochs in [25, 50, 100]:
                print('EPOCHS:', epochs)
                params = get_yolo_params(epochs=epochs, model=model)
                train_yolo(**params)

    # elif net == 'yolact':
    #     params = get_yolact_params()
    #     yolact_training(**params)


if __name__ == '__main__':
    main()

# py train.py --img 256 --batch 32 --epochs 50 --data sp_dataset.yaml --weights yolov5m.pt (To train)