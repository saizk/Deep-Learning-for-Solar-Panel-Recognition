import os
import cv2
import glob
import torch
import numpy as np
import skimage.measure as km
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from skimage import io
from skimage.filters import threshold_otsu
from torchvision.ops import nms
from concurrent.futures import ThreadPoolExecutor


__all__ = ['create_yolo_annotation']


def load_data(path):
    with ThreadPoolExecutor(max_workers=64) as pool:
        images = pool.map(
            lambda img: cv2.imread(f'{path}/{img}', 0),
            os.listdir(path)
        )
    return list(images)


def draw_bbox(img_path, bboxes):
    image = io.imread(img_path)
    for box in bboxes:
        cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.imshow(image)
    ax1.set_title('Mask')
    ax2.imshow(image)
    plt.show()


def save_paths(images_path, filename):
    paths = [str(Path(i)) + '\n' for i in glob.glob(f'{images_path}/*')]
    with open(filename, "w") as f:
        f.writelines(paths)


def normalize_bbox(bbox, size=(256, 256)):
    img_w, img_h = size
    x = 0.5 * (bbox[1] + bbox[3]) / img_w
    y = 0.5 * (bbox[0] + bbox[2]) / img_h
    w = (bbox[3] - bbox[1]) / img_w
    h = (bbox[2] - bbox[0]) / img_h
    return x, y, w, h


def gen_bboxes(masks_path, normalize=True):
    # bboxes = [get_bbox(mask, False) for mask in load_data(masks_path)]
    # for imgname, box in zip(glob.glob('solar_panels/test/images/*.png'), bboxes):
    #     print(imgname, len(box))
    #     if len(box) > 25:  # NMS!
    #         draw_bbox(imgname, box)  # normalize = False
    return [(0, get_bbox(mask, normalize)) for mask in load_data(masks_path)]


def get_bbox(mask, normalize):
    binary = mask > threshold_otsu(mask)
    labeled = km.label(binary)
    props = km.regionprops(labeled)
    bboxes = set([p.bbox for p in props])
    if normalize:
        bboxes = list(map(normalize_bbox, bboxes))
    return bboxes


def save_labels(img_path, masks_path, labels_path, normalize=True):
    img_files = os.listdir(img_path)
    boxes = gen_bboxes(masks_path, normalize)
    if not labels_path.exists():
        os.mkdir(labels_path)

    for image, (cls, box_coords) in zip(img_files, boxes):
        img_name = f'{labels_path}/{Path(image).stem}'
        with open(f'{img_name}.txt', "w") as f:
            for box in box_coords:
                f.write(f'{cls} ')
                for coord in box:
                    f.write(f'{coord} ')
                f.write('\n')


def create_yolo_annotation(images_path='images', masks_path='masks', filename='phase.txt'):
    labels_path = Path(f'{Path(images_path).parent}/labels')

    save_paths(images_path, filename)
    save_labels(images_path, masks_path, labels_path)
