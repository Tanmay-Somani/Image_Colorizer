import numpy as np
import cv2
import streamlit as st
from PIL import Image
import io

def colorizer(img):
    prototxt = r"C:\Users\Tanmay Somani\OneDrive\Desktop\Programming\!Projects\Imagecolorizer\models\models_colorization_deploy_v2.prototxt"
    model = r"C:\Users\Tanmay Somani\OneDrive\Desktop\Programming\!Projects\Imagecolorizer\models\colorization_release_v2.caffemodel"
    points = r"C:\Users\Tanmay Somani\OneDrive\Desktop\Programming\!Projects\Imagecolorizer\models\pts_in_hull.npy"
    
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    
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
    colorized = (255 * colorized).astype("uint8")
    
    return colorized

def sharpen_image(img):
    gaussian_blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)
    return sharpened

st.set_page_config(page_title="Image Colorizer", layout="centered")

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
        st.image(color, use_container_width=True)
        
        st.subheader("Sharpened Colorized Image")
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
    st.write("**Projects:** Book Recommender, Shark Tank India Dashboard, Facial Recognition System")
    st.write("Feel free to connect!")