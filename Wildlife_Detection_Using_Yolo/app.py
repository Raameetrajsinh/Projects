import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import os



model_path = os.path.join(os.path.dirname(__file__), "best.pt")

@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

st.title("Wildlife Detection")

with st.sidebar:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    frame_interval = st.number_input("No. of Frames", min_value=1, max_value=60, value=10, step=1)
    process_button = st.button("Detect")

if uploaded_video and process_button:
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(uploaded_video.read())

    cap = cv2.VideoCapture(input_temp.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()
    frame_count = 0

    with st.spinner("Processing video..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                results = model.predict(source=frame, conf=0.3, imgsz=640)
                annotated = results[0].plot()
                out.write(annotated)
                stframe.image(annotated, channels="BGR", use_container_width=True)
            else:
                out.write(frame)

            frame_count += 1

    cap.release()
    out.release()

    st.success("Video Ended.")
