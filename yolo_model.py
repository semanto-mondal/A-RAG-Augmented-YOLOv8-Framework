from ultralytics import YOLO
from PIL import Image
import streamlit as st

@st.cache_resource
def load_yolo_model(weights_path):
    model = YOLO(weights_path)
    return model

def detect_diseases(image: Image.Image, model):
    results = model(image)
    names = results[0].names
    boxes = results[0].boxes
    detected_classes = [names[int(cls)] for cls in boxes.cls.cpu()]
    annotated_img = results[0].plot()
    return detected_classes, annotated_img
