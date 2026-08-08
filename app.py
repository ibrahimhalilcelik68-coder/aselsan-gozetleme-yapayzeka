import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

st.set_page_config(page_title="ASELSAN Gözetleme AI", layout="wide")
st.title("🔭 ASELSAN Gözetleme Yapay Zeka Demo")
st.write("Kameradan canlı olarak insan, araç ve hayvan tespiti yapar.")

# Modeli yükle
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt') # küçük ve hızlı model
    return model

model = load_model()

# Video kaynağı seç
secenek = st.radio("Kaynak Seç:", ("Webcam", "Video Yükle"))

if secenek == "Webcam":
    run = st.button('Kamerayı Başlat')
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=0.5)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
        cap.release()
else:
    video_file = st.file_uploader("Video yükle", type=['mp4', 'avi'])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=0.5)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
        cap.release()
