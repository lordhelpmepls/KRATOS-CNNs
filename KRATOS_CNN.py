#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:30:05 2025

@author: marlinmahmud
"""

### TESTED - NOT OK

import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load image paths and labels
# Labels are based on file names (e.g., 'face_on' -> 0, 'edge_on' -> 1)

image_folder = "density_maps/"
image_paths = glob.glob(image_folder + "*.png")

labels = []
for path in image_paths:
    if "face_on" in path:
        labels.append(0)  # Label for 'face_on'
    elif "edge_on" in path:
        labels.append(1)  # Label for 'edge_on'

# Split data into train, validation, and test Sets
x_train_paths, x_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
x_val_paths, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Load and Preprocess Images
def load_images(image_paths, target_size=(128, 128)):
    images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=target_size)  # Resize images
        img_array = image.img_to_array(img)  # Convert to array
        images.append(img_array)
    return np.array(images)

# Load and preprocess images
x_train = load_images(x_train_paths)
x_val = load_images(x_val_paths)
x_test = load_images(x_test)

# Normalize images to [0, 1]
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# One-hot encode labels (for classification)
y_train = to_categorical(y_train, num_classes=2)  
y_val = to_categorical(y_val, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # 3 channels for RGB
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Two output classes for classification
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # For classification
              metrics=['accuracy'])

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stop]
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Save 
model.save("galaxy_classification_model.h5")