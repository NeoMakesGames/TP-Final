import tensorflow as tf
import tensorflow.keras as keras
import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.title("¿Tenes anteojos?")

#Cargamos el modelo.
model = keras.models.load_model("model.h5")

#Recibe una imágen de la cámara si es que lo eligió el usuario.
camera = st.camera_input(label="Tomate una foto para ver si tenes anteojos")
#Recibe una imágen del archivo cargado si es que lo eligió el usuario.
archivo = st.file_uploader("Subi una foto si no tenes camara")

#Si el usuario eligió la cámara.
if camera:
    cap = cv2.VideoCapture(0)

    # Capturar un frame desde la cámara
    ret, frame = cap.read()

    # Liberar la cámara
    cap.release()

    # Convertir el frame a formato Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Le ponemos un tamaño de 180x180 que es con el que trabaja la IA.
    img = img.resize((180, 180))
    # Lo convertimos a un array para que la IA pueda recibir cada uno de los valores de los pixeles en la capa de entrada.
    img = np.array(img)
    # Lo pasamos a blanco y negro porque el color no aporta y solo hace que la IA tenga que procesar más.
    img = tf.image.rgb_to_grayscale(img)
    # Le agregamos una dimensión extra al modelo que representa el batch size
    img = np.expand_dims(img, axis=0)
    # Hacemos la predicción.
    prediction = model.predict(img)
    # Le informamos al usuario dependiendo del resultado.
    if prediction[0][0] > 0.5:
        st.text("No tenes anteojos")
    else:
        st.text("Tenes anteojos")

#Si eligió cargar un archivo.
if archivo:
    #Convertimos la imagen al tipo de dato Image.
    img = Image.open(archivo)
    #Le ponemos un tamaño de 180x180 que es con el que trabaja la IA.
    img = img.resize((180, 180))
    #Lo convertimos a un array para que la IA pueda recibir cada uno de los valores de los pixeles en la capa de entrada.
    img = np.array(img)
    #Lo pasamos a blanco y negro porque el color no aporta y solo hace que la IA tenga que procesar más.
    img = tf.image.rgb_to_grayscale(img)
    #Le agregamos una dimensión extra al modelo que representa el batch size
    img = np.expand_dims(img, axis=0)
    #Hacemos la predicción.
    prediction = model.predict(img)
    #Le informamos al usuario dependiendo del resultado.
    if prediction[0][0] > 0.5:
        st.text("No tenes anteojos")
    else:
        st.text("Tenes anteojos")