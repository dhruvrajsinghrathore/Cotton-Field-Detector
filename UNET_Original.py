#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 01:30:19 2024

@author: nitaishah
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import tensorflow as tf
import glob
import matplotlib.pyplot as plt

# Directory containing your images and masks
image_folder = '/Users/nitaishah/Desktop/UNET2'

def load_data(image_folder):
    images = []
    masks = []
    
    for image_path in glob.glob(os.path.join(image_folder, "*[!mask].png")):
        mask_path = image_path.replace(".png", "_mask.png")
        
        # Load and preprocess image
        image = img_to_array(load_img(image_path, color_mode="rgb")) / 255.0
        mask = img_to_array(load_img(mask_path, color_mode="grayscale")) / 255.0
        
        images.append(image)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

images, masks = load_data(image_folder)

X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

def unet_model(input_shape=(128, 128, 1)):
    inputs = tf.keras.layers.Input(input_shape)
    
    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    # Add more encoder layers if needed
    
    # Decoder
    u1 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(p1)
    u1 = tf.keras.layers.concatenate([u1, c1])
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])

model = unet_model(input_shape=X_train.shape[1:])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Instantiate and compile the model
model = unet_model(input_shape=(512, 512, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=8, epochs=10)


