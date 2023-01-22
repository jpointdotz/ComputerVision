import tensorflow as tf
from tensorflow import keras

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.layers.convolutional import Conv

batch_size = 64
epochs = 50
IMG_HEIGHT = 150 
IMG_WIDTH = 150

def train_image_generator():

    train_image_generator = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.3
    )

    return train_image_generator

def validation_image_generator():

    validation_image_generator = ImageDataGenerator(
        rescale = 1./255
    )

    return validation_image_generator

def train_data_generator(train_image_generator):

    train_data_generator = train_image_generator.flow_from_directory(batch_size = batch_size,
    directory= 'train' ,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary')

    return train_data_generator

def validation_data_generator(validation_image_generator):

    val_data_generator = validation_image_generator.flow_from_directory(batch_size=batch_size,
    directory = 'test',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary")

    return val_data_generator

def build_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters = 16,kernel_size= 3, padding='same', activation='relu', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    return model


def compile_model(model):

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()

    return model

def train_model(model,val_data_generator):

    history = model.fit_generator(
        train_data_generator,
        steps_per_epoch=1000,
        epochs = epochs,
        validation_data = val_data_generator,
        validation_steps=1000
    )

    return history

img_path='\\test\\dog125.jpg'

if __name__ == '__main__':
    train_image_generator = train_image_generator()
    val_image_generator = validation_image_generator()
    train_data_generator = train_data_generator(train_image_generator)
    val_data_generator = validation_data_generator(val_image_generator)

    model = build_model()
    compiled_model = compile_model(model)

    trained_model = train_model(compiled_model, val_data_generator)

    print(trained_model)

