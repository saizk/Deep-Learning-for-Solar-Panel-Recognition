import gc
import os

import streamlit as st
import cv2
import numpy as np
import torch
import albumentations as A
# from sklearn.metrics import jaccard_score

import segmentation_models_pytorch as smp

from pathlib import Path
from utils import *

# ---------------------------------#
# Data preprocessing and Model building

ARCHITECTURE = smp.UnetPlusPlus
BACKBONE = 'se_resnext101_32x4d'
EPOCHS = 50
DEVICE = 'cpu'
model_path = f'./models/{ARCHITECTURE.__name__.lower()}_{BACKBONE}_model_{EPOCHS}.pth'
CLASSES = ['solar_panel']
preprocess_input = smp.encoders.get_preprocessing_fn(BACKBONE)
n_classes = 1
activation = 'sigmoid'

img_dir = './src/data/test/images'
mask_dir = './src/data/test/masks'
img_files = os.listdir(img_dir)
mask_files = os.listdir(mask_dir)

test_files = [os.path.join(img_dir, img) for img in img_files]

file_gts = {
    os.path.join(img_dir, img): 'Zenodo' for img in img_files
}


def use_image_from_test():
    if uploaded_file is None:
        with st.sidebar.header('2. Or use an image from our test set'):
            img_path = st.sidebar.selectbox(
                'Select an image',
                test_files,
                format_func=lambda x: f'{x.split("/")[-1]} ({(file_gts.get(x) or "None")})',
                index=1,
            )
            mask_path = img_path.replace('.png', '_label.png').replace(img_dir, mask_dir)
            # mask_path = os.path.join(mask_dir, mask_name)
            print(mask_path)
            if not os.path.exists(mask_path):
                mask_path = 'None'
        return img_path, mask_path

    else:
        st.sidebar.markdown("Remove the file above first to use our images.")


# Formatting ---------------------------------#
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
Detect solar panels from satellite images in just one click!

**You could upload your own image!**
-------
""".strip())

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
with st.sidebar.header('1. Upload your image'):
    uploaded_file = st.sidebar.file_uploader("Upload your image in png format", type=["png"])

st.sidebar.markdown("")

# img_path, mask_path = use_image_from_test()
if uploaded_file is None:
    with st.sidebar.header('2. Or use an image from our test set'):
        img_path = st.sidebar.selectbox(
            'Select an image',
            test_files,
            format_func=lambda x: f'{x.split("/")[-1]} ({(file_gts.get(x) or "None")})',
            index=1,
        )
        mask_path = img_path.replace('.png', '_label.png').replace(img_dir, mask_dir)
        # mask_path = os.path.join(mask_dir, mask_name)
        print(mask_path)
        if not os.path.exists(mask_path):
            mask_path = 'None'

else:
    st.sidebar.markdown("Remove the file above first to use our images.")


st.sidebar.markdown("""
###
### Developers:

- Daniel De Las Cuevas Turel
- Ricardo Chavez Torres
- Zijun He
- Sergio Aizcorbe Pardo
- Sergio Hidalgo López

""")


# ---------------------------------#
# Main panel

# define network parameters


def deploy1(uploaded_file, device='cpu'):
    # create model
    # model = get_model(ARCHITECTURE, BACKBONE, n_classes, activation)
    model = torch.load(model_path, map_location=device)
    # model = get_model(model_path)

    # st.write(uploaded_file)
    col1, col2, col3, col4 = st.columns((0.4, 0.4, 0.3, 0.3))

    with col1:  # visualize
        st.subheader('1.Visualize Image')
        with st.spinner(text="Loading the image..."):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            uploaded_file = cv2.imdecode(file_bytes, 1)
            image = cv2.resize(uploaded_file, (256, 256))

            st.image(
                image,
                caption='Image Uploaded')

    with col2:  # classify
        st.subheader('2. Model Prediction')
        with st.spinner(text="The model is running..."):
            img = imgread_preprocessing(uploaded_file, preprocess_input)
            # image = np.expand_dims(img, axis=0)
            image = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
            pr_mask = model.predict(image)
            pr_mask = (pr_mask.squeeze().numpy().round())
            prob = model(image).detach().numpy()
            ann = np.argmax(prob)
            print(prob)
            print(ann)
            st.image(pr_mask, caption='Predicted Mask')

    with col3:
        st.subheader('3. Related Data')
        st.write(
            """
            **Area predicted**:...

            **Coordinates**:...
            """
        )

    del model
    gc.collect()


def deploy2(img_path, mask_path, device='cpu'):
    # model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    # model.load_weights(f'{model_path}')

    # model = get_model(ARCHITECTURE, BACKBONE, n_classes, activation)
    model = torch.load(model_path, map_location=device)

    col1, col2, col3, col4 = st.columns((0.6, 0.6, 0.6, 0.6))

    with col1:  # visualize
        st.subheader('Visualize Image')
        with st.spinner(text="Loading the image..."):
            # print(img_path)
            selected_img = cv2.imread(img_path)
            image = cv2.resize(selected_img, (256, 256))

            st.image(
                image,
                caption='Image Selected')

    with col2:  # classify
        st.subheader('Model Prediction')
        with st.spinner(text="The model is running..."):
            img = imgread_preprocessing(selected_img, preprocess_input)
            # image = np.expand_dims(img, axis=0)
            image = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
            pr_mask = model.predict(image)
            pr_mask = (pr_mask.squeeze().numpy().round())
            # gt_mask_dir = img_path.replace('.png', '_label.png')
            st.image(pr_mask, caption='Predicted Mask')
            if mask_path != 'None':
                gt_mask = mask_read_local(mask_path)
                print("GT", gt_mask.shape)
                print(gt_mask)

                intersection = np.logical_and(pr_mask, gt_mask)
                union = np.logical_or(pr_mask, gt_mask)
                iou_score = np.round(np.sum(intersection) / np.sum(union), 2)
                print('IoU is %s' % iou_score)
                # st.image(pr_mask, caption='Predicted Mask')
                st.image(gt_mask, caption='Ground Truth Mask')

    with col3:
        st.subheader('Model Performance')
        st.write(f'**IoU score**: {iou_score}')
        del model
        gc.collect()

    with col4:
        st.subheader('Other information')
        st.write(
            """
            **Area predicted**:...

            **Coordinates**:...
            """
        )


if uploaded_file is not None:
    deploy1(uploaded_file, device=DEVICE)
elif img_path is not None:
    print(mask_path)
    deploy2(img_path, mask_path, device=DEVICE)
