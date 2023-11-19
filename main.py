import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

def convert_to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

# Load and split the dataset
data_dir = 'Caras/'
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size).map(convert_to_grayscale)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size).map(convert_to_grayscale)

outside_ds = tf.keras.utils.image_dataset_from_directory(
    "Outside/",
    image_size=(180, 180),
    batch_size=32).map(convert_to_grayscale)

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

model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

epochs = 5
history = model.fit(
    train_ds,
    epochs=epochs
)

loss, acc = model.evaluate(val_ds)
print("Accuracy", acc)

model.save('model.h5')