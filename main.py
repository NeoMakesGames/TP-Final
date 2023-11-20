# Creación, Entrenamiento y Validación --> Modelo de IA que detecta si alguien esta usando o no anteojos.

# Importamos las librerías
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Función para convertir las imágenes a escala de grises.
def convert_to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

# Cargamos los datos, seteamos la resolución de las imagenes y el tamaño del batch.
data_dir = 'Caras/'
batch_size = 32
img_height = 180
img_width = 180

# Seteamos los datos de entrenamiento y validación:
# Dataset del training.
train_ds = tf.keras.utils.image_dataset_from_directory(
    # Directorio de los datos.
    data_dir,
    # Especificamos el split.
    validation_split=0.2,
    # Especificamos el subset: 'training'.
    subset="training",
    seed=123,
    # Especificamos el tamaño de las imagenes.
    image_size=(img_height, img_width),
    # Especificamos el tamaño del batch. Convierte el dataset a 'grayscale'.   
    batch_size=batch_size).map(convert_to_grayscale)

# Dataset de validation.
val_ds = tf.keras.utils.image_dataset_from_directory(
    # Directorio de los datos.
    data_dir,
    # Especificamos el split.
    validation_split=0.2,
    # Especificamos el subset: 'training'.
    subset="validation",
    seed=123,
    # Especificamos el tamaño de las imagenes.
    image_size=(img_height, img_width),
    # Especificamos el tamaño del batch. Convierte el dataset a 'grayscale'.   
    batch_size=batch_size).map(convert_to_grayscale)

# Arquitectura del modelo:
model = models.Sequential([
    layers.Conv2D(8, (3,3), padding='same', activation='relu', input_shape=(180, 180, 1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(16, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(32, activation='relu'),

    layers.Dropout(0.75),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compilamos el modelo.
model.compile(
    # Especificamos un optimizador: 'Adam()'. 
    optimizer='adam',
    # Especificamos una función de loss, 
    loss=tf.keras.losses.binary_crossentropy,
    # Seteamos 'accuracy' como metrica de monitoreo
    metrics=['accuracy']
)

#Training del Modelo.
#Seteamos los epochs.
epochs = 5
history = model.fit(
    # Especificamos el dataset de entrenamiento.
    train_ds,
    # Especificamos el número de epochs.
    epochs=epochs
)

# Evaluamos el modelo con los datos de validation.
loss, acc = model.evaluate(val_ds)
# Guardamos la precisión y la loss del modelo, y mostramos la precisión.
print("Accuracy", acc)

# Guardamos el modelo en un archivo '.h5'.
model.save('model.h5')