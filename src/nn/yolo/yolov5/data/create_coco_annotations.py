import os
import cv2
import json
import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import skimage.measure as km

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pycocotools.mask as cocomask

__all__ = ['create_coco_annotation']


def load_data(path):
    with ThreadPoolExecutor(max_workers=64) as pool:
        images = pool.map(
            lambda img: cv2.imread(f'{path}/{img}', 0),
            os.listdir(path)
        )

    return list(images)


def create_info(description, url, year):
    return {
        "description": description,
        "url": url,
        "version": "1.0",
        "year": year,
        "date_created": datetime.datetime.now().strftime("%d/%m/%Y")
    }


def create_categories(categories):
    return [{
        "id": _id,
        "name": category,
        "supercategory": super_category
    } for _id, category, super_category in categories]


def create_images(indexes, images_path, size):
    return [{
        "id": indexes[idx] + 1,
        "file_name": filename,  # int
        "width": size[0],  # list
        "height": size[0],  # int
    } for idx, filename in enumerate(os.listdir(images_path))]


def get_segmentation(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        if contour.size >= 6:  # Valid polygons have >= 6 coordinates (3 points)
            segmentation.append(contour.flatten().tolist())

    return segmentation


def normalize_bbox(bbox, size=(256, 256)):
    img_w, img_h = size
    x = 0.5 * (bbox[1] + bbox[3]) / img_w
    y = 0.5 * (bbox[0] + bbox[2]) / img_h
    w = (bbox[3] - bbox[1]) / img_w
    h = (bbox[2] - bbox[0]) / img_h
    return x, y, w, h


def get_bbox(mask, normalize):
    labeled = km.label(mask)
    props = km.regionprops(labeled)
    bboxes = set([p.bbox for p in props])
    if normalize:
        bboxes = list(map(normalize_bbox, bboxes))
    return bboxes


def get_annotation(mask, normalize=True):
    category_id = 1

    segmentation = get_segmentation(mask)
    bbox = get_bbox(mask, normalize)

    if not segmentation:
        category_id = 0
        return None

    RLEs = cocomask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
    RLE = cocomask.merge(RLEs)
    area = float(cocomask.area(RLE))

    return segmentation, bbox, area, category_id


def create_annotations(indexes, masks, is_crowd=0):
    annotations = [get_annotation(mask) for mask in masks]
    annotations = [ann for ann in annotations if ann]

    return [{
        "id": indexes[idx] + 1,  # int
        "image_id": indexes[idx] + 1,  # int
        "segmentation": segmentation,  # list
        "area": area,  # int
        "bbox": bbox,  # list
        "category_id": category_id,
        "is_crowd": is_crowd
    } for idx, (segmentation, bbox, area, category_id) in enumerate(annotations)]


def create_coco_annotation(indexes, images_path='images', masks_path='masks', filename='annotations.json'):
    information = create_info(
        description="Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery",
        url="https://zenodo.org/record/5171712", year=2021
    )
    images = create_images(indexes, images_path, size=(256, 256))

    categories = create_categories(((0, "background", "background"),
                                    (1, "solar_panel", "solar_panel")))
    annotations = create_annotations(indexes, load_data(masks_path))
    coco_annotation = {
        "info": information,
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    with open(filename, "w") as f:
        json.dump(coco_annotation, f)
