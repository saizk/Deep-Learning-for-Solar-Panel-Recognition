import os
import glob

import numpy as np
from scipy.spatial import distance


import yolov5.detect as yolo_eval
# import nn.yolact.yolact.train as yolact_train


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
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou


def compute_jaccard(box_a, box_b):
    return distance.jaccard(box_a, box_b)


def compute_labels_accuracy():
    for label in glob.glob('yolov5/data/solar_panels/test/labels/*.txt'):
        print(label)
        hi = np.loadtxt(label)
        print(len(hi))
        # for h in hi:
        #     print(h)
        # exit()
        # with open(label) as f:
        #     for line in f.readlines():
        #         bbox = line.strip().split(',')
        #         print(bbox)
        #         exit()
        #     # print([(*line) for line in f.readlines()])
        #     exit()
    # real_labels = 'yolov5/solar_panels/test/labels'
    # for lpath in glob.glob('results/*/labels'):
    #     print(lpath)


def yolo_testing(**kwargs):
    yolo_eval.run(**kwargs)


def get_yolo_params(name, model='best.pt'):
    return {
        'source': 'yolov5/data/solar_panels/test/images',
        'weights': model,
        'imgsz': [256, 256],
        'save_txt': True,
        'project': 'results',
        'name': name,
        'device': 'cuda:0'
    }


def main(net='yolo'):

    # if net == 'yolo':
    #     for mname in os.listdir('models'):
    #         params = get_yolo_params(mname, model=f'models/{mname}/weights/best.pt')
    #         yolo_testing(**params)
    compute_labels_accuracy()


if __name__ == '__main__':
    main()
