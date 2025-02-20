import numpy as np
import cv2
import streamlit as st
from PIL import Image
import io
import os
import urllib.request

# Set page configuration at the beginning
st.set_page_config(page_title="Image Colorizer", layout="centered")

MODEL_URL = "https://github.com/richzhang/colorization/raw/master/colorization_release_v2.caffemodel"
PROTO_URL = "https://github.com/Tanmay-Somani/Image_Colorizer/raw/main/models/models_colorization_deploy_v2.prototxt"
PTS_URL = "https://github.com/Tanmay-Somani/Image_Colorizer/raw/main/models/pts_in_hull.npy"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
PROTO_PATH = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
PTS_PATH = os.path.join(MODEL_DIR, "pts_in_hull.npy")

os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url, path):
    if not os.path.exists(path):
        st.write(f"Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        st.write("Download complete!")

download_file(MODEL_URL, MODEL_PATH)
download_file(PROTO_URL, PROTO_PATH)
download_file(PTS_URL, PTS_PATH)

def colorizer(img):
    try:
        net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
        pts = np.load(PTS_PATH)

        layer_names = net.getLayerNames()
        class8 = layer_names.index("class8_ab") + 1
        conv8 = layer_names.index("conv8_313_rh") + 1

        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        scaled = img.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        return (255 * colorized).astype("uint8")
    except Exception as e:
        st.error(f"Error in colorization: {e}")
        return img

def sharpen_image(img):
    gaussian_blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)

# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

if page == "Home":
    st.title("Black & White Image Colorizer")
    st.write("Convert your grayscale images into colorized versions effortlessly.")
    
    file = st.file_uploader("Upload a grayscale image (JPG/PNG)", type=["jpg", "png"])
    
    if file:
        image = Image.open(file)
        img = np.array(image)
        
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
        st.subheader("Colorized Image")
        color = colorizer(img)
        sharpened = sharpen_image(color)
        st.image(sharpened, use_container_width=True)
        
        buffered = io.BytesIO()
        result_image = Image.fromarray(sharpened)
        result_image.save(buffered, format="PNG")
        st.download_button(label="Download Colorized Image", data=buffered.getvalue(), file_name="colorized_image.png", mime="image/png")
        
        st.success("Colorization and Sharpening Complete!")

elif page == "About":
    st.title("About the Image Colorizer App")
    st.write("This app was developed to colorize black and white images using deep learning.")
    st.write("It utilizes OpenCV's DNN module and a pre-trained model to restore color to grayscale images.")
    st.write("### Developer Information")
    st.write("**Name:** Tanmay Somani")
    st.write("**Background:** Data Science Student at DAVV University")
    st.write("**Skills:** Python, Machine Learning, Computer Vision, Cloud Computing")
    st.write("Feel free to connect!")

