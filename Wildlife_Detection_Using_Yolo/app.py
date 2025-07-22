import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import os

model_path = os.path.join(os.path.dirname(__file__), "best.pt")
st.cache_resource.clear()

@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

st.title("Wildlife Detection")

with st.sidebar:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    frame_interval_user = st.number_input("Process Frame", min_value=1, max_value=30, value=5)

if uploaded_video is not None:
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(uploaded_video.read())
    input_temp.close()

    cap = cv2.VideoCapture(input_temp.name)

    stframe = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval_user == 0:
            results = model.predict(frame, conf=0.3, imgsz=224, device="cpu")
            annotated = results[0].plot()
        else:
            annotated = frame

        if frame_count % (frame_interval_user * 2) == 0:
            stframe.image(annotated, channels="BGR", use_container_width=True)

        frame_count += 1

    cap.release()
    st.success("Video finished.")
