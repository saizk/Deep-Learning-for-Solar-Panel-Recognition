import cv2
import numpy as np
import albumentations as A
import skimage.measure as km
import streamlit as st


@st.cache(allow_output_mutation=False, ttl=24 * 60 * 60)
def get_model(model, backbone, n_classes, activation):
    return model(backbone, classes=n_classes, activation=activation)


@st.cache(allow_output_mutation=True)
def get_test_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(256, 256),
        A.PadIfNeeded(256, 256)
    ]
    return A.Compose(test_transform)


@st.cache(allow_output_mutation=True)
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


@st.cache(allow_output_mutation=True)
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


@st.cache(allow_output_mutation=True)
def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou_score = np.round(np.sum(intersection) / np.sum(union), 2)
    return iou_score


@st.cache(allow_output_mutation=True)
def compute_pixel_acc(gt_mask, pred_mask):
    total = 256 ** 2
    errors = (np.abs((pred_mask == 0).sum() - (gt_mask == 0).sum()) + np.abs(
        (pred_mask != 0).sum() - (gt_mask != 0).sum()))
    accuracy = np.round(1 - (errors / total), 3)
    return accuracy


@st.cache(allow_output_mutation=True)
def compute_dice_coeff(gt_mask, pred_mask):
    intersection = np.sum(np.logical_and(pred_mask, gt_mask))
    union = np.sum(np.logical_or(pred_mask, gt_mask)) + intersection
    dice_coef = np.round(2 * intersection / union, 3)
    return dice_coef


@st.cache(allow_output_mutation=True)
def compute_bboxes(image, pred_mask):
    """

    :param image:
    :param pred_mask:
    :return:
    """
    kernel = np.ones((5, 5), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    labeled = km.label(pred_mask)
    props = km.regionprops(labeled)
    bboxes = set([p.bbox for p in props])
    return bboxes


@st.cache(allow_output_mutation=True)
def draw_bbox(image, bboxes):
    for box in bboxes:
        image = cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
    return image


@st.cache(allow_output_mutation=True)
def return_coordinates(bboxes):
    num_boxes = len(bboxes)
    coordinate_dict = {}
    for i in range(num_boxes):
        box = list(bboxes)[i]
        top_left = (box[1], box[0])
        bottom_right = (box[3], box[2])
        top_right = (box[3], box[0])
        bottom_left = (box[1], box[2])
        coordinate_dict[i] = [top_left, top_right, bottom_left, bottom_right]
    return coordinate_dict


def get_model_info(model_name):
    info, ext = model_name.split('.')
    arch, *enc, epochs = info.split('_')

    enc = '_'.join(enc[:-1])
    return arch, enc, int(epochs)
