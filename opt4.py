import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

st.set_page_config(
    page_title="StyleShift",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("StyleShift")
st.markdown("Create Digital Art using Machine Learning! We take 2 images - Content Image & Style Image - and blend them together so that the resulting output image retains the core elements of the content image, but appears to be 'painted' in the style of the style reference image.")

def load_image(image, max_dim):
    img = Image.open(image)
    img = img.convert("RGB")
    img = np.array(img)
    img = img.astype(np.float32)[np.newaxis, ...] / 255.
    max_dim = max_dim
    shape = tf.cast(tf.shape(img)[1:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = tf.image.convert_image_dtype(img, tf.float32) 
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.cast(tensor, tf.uint8)
    tensor = tensor.numpy()
    return Image.fromarray(tensor[0])

hub_module = hub.load('https://www.kaggle.com/models/google/arbitrary-image-stylization-v1/TensorFlow1/256/2')

col1, col2, col3 = st.columns([1, 0.1, 1])

with col1:
    st.markdown("## Content Image")
    content_image = st.file_uploader("Drag and drop file here", type=["png", "jpg"], key="content_image")
    if content_image:
        content_img = load_image(content_image, 1000)
        st.image(tensor_to_image(content_img))

with col3:
    st.markdown("## Style Image")
    style_image = st.file_uploader("Drag and drop file here", type=["png", "jpg"], key="style_image")
    if style_image:
        style_img = load_image(style_image, 1000)
        st.image(tensor_to_image(style_img))

if content_image and style_image:
    stylized_image = hub_module(tf.constant(content_img, dtype=tf.float32), tf.constant(style_img, dtype=tf.float32))[0]
    st.markdown("## Stylized Image")
    st.image(tensor_to_image(stylized_image))

st.markdown("## Some Inspiration ")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image(Image.open('image_1.jpg'))
with col2:
    st.image(Image.open('image_2.jpg'))
with col3:
    st.image(Image.open('image_3.jpg'))

st.markdown("---")