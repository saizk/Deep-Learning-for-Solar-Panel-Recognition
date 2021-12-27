import os
import glob
import torch

import numpy as np
from scipy.spatial import distance

import yolov5.detect as yolo_eval


def compute_iou_bb(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
    if not inter_area:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    box_b_area = abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou


def compute_jaccard(box_a, box_b):
    return distance.jaccard(box_a, box_b)


def compute_labels_accuracy():
    real_labels = glob.glob('yolov5/data/solar_panels/test/labels/*.txt')
    predicted_labels = glob.glob('results/yolov5l_100/labels/*.txt')
    # predicted_labels = glob.glob('results/*/labels/*.txt')
    r_sum, p_sum = 0, 0
    for label in real_labels:
        for plabel in predicted_labels:
            if label.split('/')[-1] == plabel.split('/')[-1]:
                real, pred = np.loadtxt(label), np.loadtxt(plabel)
                print(f'R:{len(real)} P:{len(pred)} -> {label.split("/")[-1]} {"!!" if len(real) != len(pred) else ""}')
                r_sum += len(real)
                p_sum += len(pred)
    print(r_sum)
    print(p_sum)
    print(p_sum/r_sum)


def yolo_testing(**kwargs):
    yolo_eval.run(**kwargs)


def get_yolo_params(name, project='results', model='best.pt', device='cuda:0'):
    return {
        # 'source': 'yolov5/data/solar_panels/test/images',
        'source': 'yolov5/data/solar_panels/tiles2',
        'weights': model,
        'imgsz': [256, 256],
        'save_txt': True,
        'project': project,
        'name': name,
        'evolove': 300,
        'line_thickness': 2,
        'hide_labels': True,
        'device': device
    }


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # for mname in os.listdir('models'):
    #     params = get_yolo_params(mname, project='results',
    #                              model=f'models/{mname}/weights/best.pt', device=device)
    #     yolo_testing(**params)
    mname = 'yolov5x_100'
    params = get_yolo_params(f'{mname}_tiles2', project='results',
                             model=f'models/{mname}/weights/best.pt', device=device)
    yolo_testing(**params)

    # compute_labels_accuracy()


if __name__ == '__main__':
    main()
