# app.py
import streamlit as st
import numpy as np
import cv2
from detector import detect_and_estimate

st.set_page_config(page_title="Object Detection + Distance Estimation", layout="wide")
st.title("🚗 Object Detection + Distance Estimation for Autonomous Vehicles")

model_choice = st.selectbox("Select Detection Model:", ["YOLOv8", "YOLOv5"])
input_type = st.radio("Choose Input Source:", ["Image Upload", "Webcam"])

if input_type == "Image Upload":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, 1)
        result = detect_and_estimate(img, model_name=model_choice)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Image", use_column_width=True)

elif input_type == "Webcam":
    run = st.checkbox("Start Webcam")
    frame_window = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            result = detect_and_estimate(frame, model_name=model_choice)
            frame_window.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()
