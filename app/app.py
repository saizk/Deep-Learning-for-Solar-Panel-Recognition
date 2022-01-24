import gc
import os
import copy
import sklearn
import cv2
import torch
import numpy as np
import albumentations as A
import skimage.measure as km
import streamlit as st
from pathlib import Path
# from sklearn.metrics import jaccard_score
from matplotlib import pyplot as plt

import segmentation_models_pytorch as smp

# ---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(
    page_title='Solar Panels Detection',
    # anatomical heart favicon
    page_icon="https://api.iconify.design/openmoji/solar-energy.svg?width=500",
    layout='wide'
)

# PAge Intro
st.write("""
# :sunny: Solar Panel Detection
Detect solar panels from satellite images with just one click!

**You could upload your own image!**
-------
""".strip())


# ---------------------------------#
# Data preprocessing and Model building

@st.cache(allow_output_mutation=True)
def imgread_preprocessing(uploaded_img):  # final preprocessing function in streamlit
    # read data
    # CLASSES = ['solar_panel']
    # class_values=[CLASSES.index(cls.lower()) for cls in classes]

    # image = cv2.imread(uploaded_img)
    image = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(uploaded_mask,0)
    # mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

    # extract certain classes from mask (e.g. cars)
    # masks = [(mask!=v) for v in class_values]
    # mask = np.stack(masks, axis=-1).astype('float')

    # add background if mask is not binary
    # if mask.shape[-1] != 1:
    #    background = 1 - mask.sum(axis=-1, keepdims=True)
    #    mask = np.concatenate((mask, background), axis=-1)

    # apply augmentations
    augmentation = get_test_augmentation()
    sample = augmentation(image=image)
    image = sample['image']

    # apply preprocessing
    preprocessing = get_preprocessing(preprocess_input)
    sample = preprocessing(image=image)
    image = sample['image']
    return image


@st.cache(allow_output_mutation=True)
def mask_read_local(gt_mask_dir):
    gt_mask = cv2.imread(gt_mask_dir, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.threshold(gt_mask, 0, 255, cv2.THRESH_BINARY)[1]
    gt_mask = cv2.resize(gt_mask, (256, 256))
    return gt_mask


@st.cache(allow_output_mutation=True)
def mask_read_uploaded(uploaded_mask):
    file_bytes = np.asarray(bytearray(uploaded_mask.read()), dtype=np.uint8)
    uploaded_mask = cv2.imdecode(file_bytes, 1)
    uploaded_mask = cv2.resize(uploaded_mask, (256, 256))
    uploaded_mask = uploaded_mask[:, :, 0]
    uploaded_mask = cv2.threshold(uploaded_mask, 0, 255, cv2.THRESH_BINARY)[1]
    return uploaded_mask


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
def compute_dice_coef(gt_mask, pred_mask):
    intersection = np.sum(np.logical_and(pred_mask, gt_mask))
    union = np.sum(np.logical_or(pred_mask, gt_mask)) + intersection
    dice_coef = np.round(2 * intersection / union, 3)
    return dice_coef


@st.cache(allow_output_mutation=True)
def show_detection(image, pred_mask):
    """

    :param image: original image
    :param pred_mask: predicted binary mask
    :return: original image with detected solar panels colored
    """

    pred_mask = cv2.threshold(pred_mask, 0, 255, cv2.THRESH_BINARY)[1]

    pred_mask = np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2)
    result = cv2.bitwise_and(image.astype('int'), pred_mask.astype('int'))

    return result


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


# Formatting ---------------------------------#

hide_streamlit_style = """
        <style>
        MainMenu {visibility: hidden;}
        footer {	
            visibility: hidden;
        }
        footer:after {
            content:'Created with Streamlit';
            visibility: visible;
            display: block;
            position: relative;
            #background-color: grey;
            #primary-color: blue;
            padding: 5px;
            top: 2px;
        }
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------#
# Sidebar - Collects user input features into dataframe

with st.sidebar.header('Upload your image to detect solar panels'):
    uploaded_file = st.sidebar.file_uploader("Upload your image in png format", type=["png"])

if uploaded_file is not None:
    with st.sidebar.header('You can upload its ground truth mask to compute scores'):
        uploaded_mask = st.sidebar.file_uploader("Upload its binary mask in png format", type=["png"])

st.sidebar.markdown("")

img_dir = os.path.join('.', "data")
img_files = list(filter(lambda x: 'label' not in x, os.listdir(img_dir)))

file_gts = {
    img.replace('.png', ''): 'Zenodo'
    for img in img_files
}

if uploaded_file is None:
    with st.sidebar.header('Use an image from our test set'):
        pre_trained_img = st.sidebar.selectbox(
            'Select an image',
            img_files,
            format_func=lambda x: f'{x} ({(file_gts.get(x.replace(".png", "")))})' if ".png" in x else x,
            index=1,
        )
        if pre_trained_img != "None":
            selected_img_dir = img_dir + '/' + pre_trained_img

else:
    st.sidebar.markdown("Remove the file above first to use our images.")

# define network parameters
ARCHITECTURE = smp.UnetPlusPlus
BACKBONE = 'se_resnext101_32x4d'
CLASSES = ['solar_panel']
activation = 'sigmoid'
EPOCHS = 25
DEVICE = 'cpu'
n_classes = len(CLASSES)
preprocess_input = smp.encoders.get_preprocessing_fn(BACKBONE)

model_dir = os.path.join('.', 'models')
model_path = f'{model_dir}/{ARCHITECTURE.__name__.lower()}_{BACKBONE}_{EPOCHS}ep.pth'

models = {
    'DeeplabV3Plus': {'ARCHITECTURE': smp.DeepLabV3Plus, 'BACKBONE': 'efficientnet-b3', 'EPOCHS': 25},
    'UNET++Resnext101': {'ARCHITECTURE': smp.UnetPlusPlus, 'BACKBONE': 'se_resnext101_32x4d', 'EPOCHS': 50},
    'UNET++Resnest50d': {'ARCHITECTURE': smp.UnetPlusPlus, 'BACKBONE': 'timm-resnest50d_4s2x40d', 'EPOCHS': 50},
    'UNET++Vgg19-BN': {'ARCHITECTURE': smp.UnetPlusPlus, 'BACKBONE': 'vgg19_bn', 'EPOCHS': 50}
}

model_options = models.keys()
with st.sidebar.subheader('Select the model you want to use for prediction'):
    model_sel = st.sidebar.selectbox(
        'Select a model (architecture+backbone)',
        model_options
    )
    model = models[model_sel]
    model_path = f'{model_dir}/{model["ARCHITECTURE"].__name__.lower()}_{model["BACKBONE"]}_{model["EPOCHS"]}ep.pth'
    ARCHITECTURE = model['ARCHITECTURE']
    BACKBONE = model['BACKBONE']
    EPOCHS = model['EPOCHS']
    DEVICE = 'cpu'
    preprocess_input = smp.encoders.get_preprocessing_fn(BACKBONE)

######

st.sidebar.markdown("""
###
### Developers:

- Sergio Aizcorbe Pardo
- Ricardo Chavez Torres
- Daniel De Las Cuevas Turel
- Sergio Hidalgo López
- Zijun He

""")

# ---------------------------------#
# Main panel

def deploy1(uploaded_file, uploaded_mask=None):
    # create model
    # model = get_model(ARCHITECTURE, BACKBONE, n_classes, activation)

    model = torch.load(model_path, map_location='cpu')
    # model = get_model(model_path)

    # st.write(uploaded_file)
    col1, col2, col3, col4 = st.columns((0.4, 0.4, 0.3, 0.3))

    if uploaded_mask is not None:
        gt_mask = mask_read_uploaded(uploaded_mask)

    with col1:  # visualize
        st.subheader('1.Visualize Image')
        with st.spinner(text="Loading the image..."):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            selected_img = cv2.imdecode(file_bytes, 1)
            image = cv2.resize(selected_img, (256, 256))

        st.subheader('Visualize Image')
        detec_option = st.selectbox(label='Display options',
                                    options=['Show original image', 'Show found solar panels'])

        if detec_option == 'Show original image':

            with st.spinner(text="Showing the image..."):
                st.image(
                    image,
                    caption='Image Selected')

        elif detec_option == 'Show found solar panels':
            with st.spinner(text="Detecting solar panels..."):
                img_pre = imgread_preprocessing(image)
                image_torch = torch.from_numpy(img_pre).to(DEVICE).unsqueeze(0)
                pr_mask = model.predict(image_torch)
                pr_mask = (pr_mask.squeeze().numpy().round())
                boxes = compute_bboxes(image, pr_mask)
                image_bboxes = draw_bbox(image, boxes)
                st.image(image_bboxes, caption='Bounding boxes detected')

    with col2:  # classify
        st.subheader('Model Prediction Masks')
        with st.spinner(text="The model is running..."):
            image = cv2.resize(selected_img, (256, 256))
            img_pre = imgread_preprocessing(image)
            image_torch = torch.from_numpy(img_pre).to(DEVICE).unsqueeze(0)
            pr_mask = model.predict(image_torch)
            pr_mask = (pr_mask.squeeze().numpy().round())

            ##################################
            # Choose what to display on mask prediction

            if uploaded_mask is not None:
                options_display_detec = ['Show predicted mask', 'Show ground truth mask', 'Show both']
                mask_option = st.selectbox(label='Display options',
                                           options=options_display_detec)
            else:
                mask_option = 'Show predicted mask'

            options_display_prediction = ['Show mask in binary', 'Show mask applied to original image']

            display_pred = st.selectbox('Mask options', options_display_prediction)

            if display_pred == 'Show mask in binary':
                if mask_option == 'Show predicted mask':
                    st.image(pr_mask, caption='Predicted Mask')

                elif mask_option == 'Show ground truth mask':
                    st.image(gt_mask, caption='Ground Truth Mask')

                elif mask_option == 'Show both':
                    st.image(pr_mask, caption='Predicted Mask')
                    st.image(gt_mask, caption='Ground Truth Mask')

            elif display_pred == 'Show mask applied to original image':
                if mask_option == 'Show predicted mask':
                    colored_image = show_detection(image, pr_mask)
                    colored_image = cv2.resize(colored_image, (256, 256))
                    st.image(colored_image, caption='Detection applied to image')

                elif mask_option == 'Show ground truth mask':
                    mask = cv2.resize(gt_mask, (256, 256))
                    mask_applied = show_detection(image, mask)
                    mask_applied = cv2.resize(mask_applied, (256, 256))
                    st.image(mask_applied, caption='Ground Truth Mask applied')

                elif mask_option == 'Show both':
                    colored_image = show_detection(image, pr_mask)
                    colored_image = cv2.resize(colored_image, (256, 256))
                    st.image(colored_image, caption='Detection applied to image')
                    mask_applied = show_detection(image, gt_mask)
                    mask_applied = cv2.resize(mask_applied, (256, 256))
                    st.image(mask_applied, caption='Ground Truth Mask applied')

    with col3:
        st.subheader('Model Performance')
        if uploaded_mask is not None:
            iou_score = compute_iou(gt_mask, pr_mask)
            st.write(f'**IoU score**: {iou_score}')
            pixel_acc = compute_pixel_acc(gt_mask, pr_mask)
            st.write(f'**Pixelwise accuracy**: {pixel_acc}')
            dice_coef = compute_dice_coef(gt_mask, pr_mask)
            st.write(f'**Dice Coefficient (F1 Score)**: {dice_coef}')
        else:
            st.write('Upload a mask if you want to check scores')

    with col4:
        st.subheader('Other information')
        spatial_res = st.text_input(label='Enter the spatial resolution of the image: how long is one pixel', value=0.1)
        area = np.round(float(spatial_res) ** 2 * pr_mask.sum(), 4)
        perc_area = np.round(100 * pr_mask.sum() / 256 ** 2, 2)
        total_area = np.round(256 ** 2 * float(spatial_res) ** 2, 2)
        st.write(
            f'**Area predicted**: {area} meters squared, which represents {perc_area}% of the image ({total_area} meters squared)')
        boxes = compute_bboxes(image, pr_mask)
        coordinates_dict = return_coordinates(boxes)
        coor_values = list(coordinates_dict.values())
        num_bboxes = len(coor_values)
        st.write(f'**Coordinates**:')
        mkd_pred_table = """
        | B. Box | Top Left | Top Right | Bottom Left | Bottom Right |
        | --- | --- | --- | --- | --- |
        """ + "\n".join(
            [f"| {i + 1} | {coor_values[i][0]} | {coor_values[i][1]} | {coor_values[i][2]} | {coor_values[i][3]} |" for
             i in range(num_bboxes)])
        st.markdown(mkd_pred_table, unsafe_allow_html=True)

    del model
    gc.collect()


def deploy2(selected_img_dir):
    # Load model
    model = torch.load(model_path, map_location='cpu')

    col1, col2, col3, col4 = st.columns((0.6, 0.6, 0.6, 0.6))

    gt_mask_dir = selected_img_dir.replace('.png', '_label.png')

    # Check if there exists GT math with "[...]_label.png"
    if os.path.isfile(gt_mask_dir):
        gt_mask = mask_read_local(gt_mask_dir)
    else:
        gt_mask = None

    selected_img = cv2.cvtColor(cv2.imread(selected_img_dir), cv2.COLOR_BGR2RGB)
    image = cv2.resize(selected_img, (256, 256))

    with col1:  # visualize
        st.subheader('Visualize Image')
        detec_option = st.selectbox(label='Display options',
                                    options=['Show original image', 'Show found solar panels'])

        if detec_option == 'Show original image':

            with st.spinner(text="Loading the image..."):
                st.image(
                    image,
                    caption='Image Selected')

        elif detec_option == 'Show found solar panels':
            with st.spinner(text="Detecting solar panels..."):
                img_pre = imgread_preprocessing(selected_img)
                image_torch = torch.from_numpy(img_pre).to(DEVICE).unsqueeze(0)
                pr_mask = model.predict(image_torch)
                pr_mask = (pr_mask.squeeze().numpy().round())
                boxes = compute_bboxes(image, pr_mask)
                image_bboxes = draw_bbox(image, boxes)
                st.image(image_bboxes, caption='Bounding boxes detected')

    with col2:  # classify
        st.subheader('Model Prediction Masks')
        with st.spinner(text="The model is running..."):
            image = cv2.resize(selected_img, (256, 256))
            img_pre = imgread_preprocessing(selected_img)
            image_torch = torch.from_numpy(img_pre).to(DEVICE).unsqueeze(0)
            pr_mask = model.predict(image_torch)
            pr_mask = (pr_mask.squeeze().numpy().round())

            if gt_mask is not None:
                options_display_detec = ['Show predicted mask', 'Show ground truth mask', 'Show both']
                mask_option = st.selectbox(label='Display options',
                                           options=options_display_detec)
            else:
                mask_option = 'Show predicted mask'

            ##################################
            # Choose what to display on mask prediction

            options_display_prediction = ['Show mask in binary', 'Show mask applied to original image']
            display_pred = st.selectbox('Mask options', options_display_prediction)

            if display_pred == 'Show mask in binary':
                if mask_option == 'Show predicted mask':
                    st.image(pr_mask, caption='Predicted Mask')

                elif mask_option == 'Show ground truth mask':
                    st.image(gt_mask, caption='Ground Truth Mask')

                elif mask_option == 'Show both':
                    st.image(pr_mask, caption='Predicted Mask')
                    st.image(gt_mask, caption='Ground Truth Mask')

            elif display_pred == 'Show mask applied to original image':
                if mask_option == 'Show predicted mask':
                    colored_image = show_detection(image, pr_mask)
                    colored_image = cv2.resize(colored_image, (256, 256))
                    st.image(colored_image, caption='Detection applied to image')

                elif mask_option == 'Show ground truth mask':
                    mask = cv2.resize(gt_mask, (256, 256))
                    mask_applied = show_detection(image, mask)
                    mask_applied = cv2.resize(mask_applied, (256, 256))
                    st.image(mask_applied, caption='Ground Truth Mask applied')

                elif mask_option == 'Show both':
                    colored_image = show_detection(image, pr_mask)
                    colored_image = cv2.resize(colored_image, (256, 256))
                    st.image(colored_image, caption='Detection applied to image')
                    mask_applied = show_detection(image, gt_mask)
                    mask_applied = cv2.resize(mask_applied, (256, 256))
                    st.image(mask_applied, caption='Ground Truth Mask applied')
    with col3:
        st.subheader('Model performance')
        if gt_mask is not None:
            iou_score = compute_iou(gt_mask, pr_mask)
            st.write(f'**IoU score**: {iou_score}')
            pixel_acc = compute_pixel_acc(gt_mask, pr_mask)
            st.write(f'**Pixelwise accuracy**: {pixel_acc}')
            dice_coef = compute_dice_coef(gt_mask, pr_mask)
            st.write(f'**Dice Coefficient (F1 Score)**: {dice_coef}')
        else:
            st.write('That image does not have a mask')

    with col4:
        st.subheader('Other information')
        spatial_res = st.text_input(label='Enter the spatial resolution of the image: how long is one pixel', value=0.1)
        area = np.round(float(spatial_res) ** 2 * pr_mask.sum(), 4)
        perc_area = np.round(100 * pr_mask.sum() / 256 ** 2, 2)
        total_area = np.round(256 ** 2 * float(spatial_res) ** 2, 2)
        st.write(
            f'**Area predicted**: {area} meters squared, which represents {perc_area}% of the image ({total_area} meters squared)')
        boxes = compute_bboxes(image, pr_mask)
        coordinates_dict = return_coordinates(boxes)
        coor_values = list(coordinates_dict.values())
        num_bboxes = len(coor_values)
        st.write(f'**Coordinates**:')
        mkd_pred_table = """
        | B. Box | Top Left | Top Right | Bottom Left | Bottom Right |
        | --- | --- | --- | --- | --- |
        """ + "\n".join(
            [f"| {i + 1} | {coor_values[i][0]} | {coor_values[i][1]} | {coor_values[i][2]} | {coor_values[i][3]} |"
             for i in range(num_bboxes)])
        st.markdown(mkd_pred_table, unsafe_allow_html=True)

    del model
    gc.collect()


if uploaded_file is not None:
    if uploaded_mask is not None:
        deploy1(uploaded_file, uploaded_mask)
    else:
        deploy1(uploaded_file)

elif pre_trained_img != 'None':
    deploy2(selected_img_dir)
