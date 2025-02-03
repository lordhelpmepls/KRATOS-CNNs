#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:44:26 2025

@author: marlinmahmud
"""

### TESTED 

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
from keras.layers import Input

# Set directory paths for image datasets
image_folder = "density_maps_K3/"
face_on_folder = os.path.join(image_folder, "face_on")
edge_on_folder = os.path.join(image_folder, "edge_on")

# Load image paths and labels
# Labels are based on folder names ('face_on' -> 0, 'edge_on' -> 1)
image_paths = glob.glob(os.path.join(face_on_folder, "*.png")) + glob.glob(os.path.join(edge_on_folder, "*.png"))
labels = []

for path in image_paths:
    if "face_on" in path:
        labels.append(0)  # Label face-on images as 0
    elif "edge_on" in path:
        labels.append(1)  # Label edge-on images as 1 

# Split data into train, validation, and test sets
x_train_paths, x_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
x_val_paths, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Load and Preprocess Images
def load_images(image_paths, target_size=(128, 128)):
    images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=target_size)  
        img_array = image.img_to_array(img)  
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

model = Sequential([
    Input(shape=(128, 128, 3)),  # Input layer defining the input shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
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
    epochs=25,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stop]
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Save 
model.save("galaxy_classification_model.h5")

from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ---- Get Predictions ---- #
y_pred_probs = model.predict(x_test)  # Get probability scores
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to class labels
y_true = np.argmax(y_test, axis=1)  # True class labels

# ---- Compute ROC Curve & AUC ---- #
fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])  # FPR, TPR for class 1
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# ---- Compute Precision-Recall Curve & AUC ---- #
precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='red', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()




import random

# Select a random image index
random_idx = random.randint(0, len(x_test) - 1)

# Get the true label
true_label = np.argmax(y_test[random_idx])  # Get true label (0 or 1)

# Get the image from x_test (already preprocessed)
random_image_array = x_test[random_idx]

# Make a prediction (add batch dimension to match model input)
random_image_array = np.expand_dims(random_image_array, axis=0)

# Make a prediction
prediction_probs = model.predict(random_image_array)
predicted_label = np.argmax(prediction_probs)

# Format prediction probabilities as percentages
predicted_prob_face_on = prediction_probs[0][0] * 100  # For class 0 (Face-on)
predicted_prob_edge_on = prediction_probs[0][1] * 100  # For class 1 (Edge-on)

# Print prediction probabilities in percentage format
print(f"Prediction Probabilities (Face-on): {predicted_prob_face_on:.2f}%")
print(f"Prediction Probabilities (Edge-on): {predicted_prob_edge_on:.2f}%")
print(f"Model Prediction: {'Face-on' if predicted_label == 0 else 'Edge-on'}")
print(f"True Label: {'Face-on' if true_label == 0 else 'Edge-on'}")





