import tensorflow as tf
import tensorflow.keras as keras
import streamlit as st
from PIL import Image
import numpy as np

st.title("¿Tenes anteojos?")

model = keras.models.load_model("model.h5")

camera = st.camera_input(label="Tomate una foto para ver si tenes anteojos")
archivo = st.file_uploader("Subi una foto si no tenes camara")

if camera:
    img = Image.fromarray(camera.astype(np.uint8))
    img = img.resize((180, 180))
    img = np.array(img)
    img = tf.image.rgb_to_grayscale(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        st.text("No tenes anteojos")
    else:
        st.text("Tenes anteojos")

if archivo:
    img = Image.open(archivo)
    img = img.resize((180, 180))
    img = np.array(img)
    img = tf.image.rgb_to_grayscale(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        st.text("No tenes anteojos")
    else:
        st.text("Tenes anteojos")