import streamlit as st
import numpy as np
import albumentations as A
import segmentation_models as sm
import cv2


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
    # image, mask = sample['image'], sample['mask']
    image = sample['image']

    # apply preprocessing
    preprocessing = get_preprocessing(preprocess_input)
    sample = preprocessing(image=image)
    # image, mask = sample['image'], sample['mask']
    image = sample['image']

    return image


model_path = '../models/best_model.h5'
BACKBONE = 'efficientnetb3'
CLASSES = ['solar_panel']
preprocess_input = sm.get_preprocessing(BACKBONE)
sm.set_framework('tf.keras')




@st.cache(allow_output_mutation=True)
def get_test_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(256, 256),
        A.PadIfNeeded(256, 256)
    ]
    return A.Compose(test_transform)


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
with st.sidebar.header('1. Upload your image'):
    uploaded_file = st.sidebar.file_uploader("Upload your image in png format", type=["png"])

st.sidebar.markdown("")

testfiles = ['None',
            'PV01_325206_1204151.png',
             'PV01_325206_1204186.png',
             'PV01_325574_1204564.png',
             'PV03_315173_1194612.png',
             'PV08_332400_1179443.png'

             ]

if uploaded_file is None:
    with st.sidebar.header('2. Or use an image from our test set'):
        pre_trained_img = st.sidebar.selectbox(
            'Select an image',
            testfiles)
        if pre_trained_img != "None":

            selected_img = "../data/test/images/" + pre_trained_img

else:
    st.sidebar.markdown("Remove the file above first to use our images.")

st.sidebar.markdown("""



### Authors:
- Sergio Aizcorbe Pardo
- Ricardo Chavez Torres
- Sergio Hidalgo López
- Daniel De Las Cuevas Turel
- Zijun He
""")
# ---------------------------------#
# Main panel

# define network parameters
n_classes = 1
activation = 'sigmoid'

# create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
model.load_weights(f'{model_path}')

if uploaded_file is not None:
    # st.write(uploaded_file)
    col1, col2, col3 = st.columns((0.4, 0.4, 0.2))

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
            img = imgread_preprocessing(uploaded_file)
            image = np.expand_dims(img, axis=0)
            pr_mask = model.predict(image).round()

            st.image(pr_mask, caption='Predicted Mask')

elif pre_trained_img !='None':
        col1, col2, col3 = st.columns((0.4, 0.4, 0.2))

        with col1:  # visualize
            st.subheader('1.Visualize Image')
            with st.spinner(text="Loading the image..."):
                selected_img = cv2.imread(selected_img)
                image = cv2.resize(selected_img, (256, 256))

                st.image(
                    image,
                    caption='Image Selected')

        with col2:  # classify
            st.subheader('2. Model Prediction')
            with st.spinner(text="The model is running..."):
                img = imgread_preprocessing(selected_img)
                image = np.expand_dims(img, axis=0)
                pr_mask = model.predict(image).round()

                st.image(pr_mask, caption='Predicted Mask')

        with col3:
            st.subheader('3. Related Data')
            st.write(
                """
                **Area predicted**:...
                
                **Coordinates**:...
                """
            )
