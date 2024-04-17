import io
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

def load_model():
    model = YOLO('./best.pt')
    return model


def load_image():
    uploaded_file = st.file_uploader(label="Выберите изображение для детекции")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
    

def detect_objects():
    results = model(img, imgsz=640, iou=0.4, conf=0.4, verbose=True)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(annotated_frame)


model = load_model()

st.set_page_config(page_title="SkyVision")
st.markdown("<h1 style='text-align: center;'>SkyVision</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align: center;'>Детекция объектов на спутниковых снимках земной поверхности</h3>", unsafe_allow_html=True)

img = load_image()

result = st.button("Запустить детекцию")

if img and result:
    st.write("### **Результаты детекции:**")
    detect_objects()
