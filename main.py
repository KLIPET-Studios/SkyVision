import io
import streamlit as st
from PIL import Image
import numpy as np

from keras.applications import EfficientNetB0
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input, decode_predictions


def load_model():
    model = EfficientNetB0(weights="imagenet")
    return model


def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_image():
    uploaded_file = st.file_uploader(label="Выберите изображение для распознования")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
    

def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(cl[1], ": " + str(round(100*cl[2], 2)) + "%")

    

model = load_model()

st.set_page_config(page_title="SkyVision")
st.title("Классификация изображений")
img = load_image()
result = st.button("Распознать изображение")
if img and result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write("**Результаты распознования:**")
    print_predictions(preds)