#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:43:44 2025

@author: marlinmahmud
"""

import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications import VGG16
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import random

image_folder = "density_maps_K3"
categories = ["barred_grand_spiral", "barred_weak_spiral", "barred_no_spiral", "no_bar_no_spiral"]

image_paths = []
labels = []

for idx, category in enumerate(categories):
    category_path = os.path.join(image_folder, category)
    images = glob.glob(os.path.join(category_path, "*.png"))
    image_paths.extend(images)
    labels.extend([idx] * len(images))  # Assign labels (0-3)

# Split data into train, validation, and test sets
x_train_paths, x_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
x_val_paths, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Load and preprocess images
def load_images(image_paths, target_size=(128, 128)):
    images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        images.append(img_array)
    return np.array(images)

x_train = load_images(x_train_paths)
x_val = load_images(x_val_paths)
x_test = load_images(x_test)

# Normalize images to [0, 1]
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# One-hot encode labels (4 categories)
y_train = to_categorical(y_train, num_classes=4)
y_val = to_categorical(y_val, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

# Data Augmentation for Training Data
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)

# Fit augmentation generator to training data
datagen.fit(x_train)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

# Load pre-trained VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model with a lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model with augmented data
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=8),
    epochs=25,
    validation_data=(x_val, y_val),
    steps_per_epoch=len(x_train) // 8,
    class_weight=class_weights_dict,
    callbacks=[reduce_lr],
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

model.save("galaxy_morphology_classification_K3_transfer_learning.keras")

# Precision-Recall Curve & AUC
y_pred_probs = model.predict(x_test)  # Probability scores
y_true = np.argmax(y_test, axis=1)  # True class labels
y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted class labels

# ROC & Precision-Recall Curves for each class
for i in range(4):
    fpr, tpr, _ = roc_curve(y_true == i, y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})', color='red')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Class {categories[i]}')
    plt.legend(loc='lower right')
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true == i, y_pred_probs[:, i])
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.2f})', color='magenta')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Class {categories[i]}')
    plt.legend(loc='lower left')
    plt.show()

# Select a random image from the test set
random_idx = random.randint(0, len(x_test) - 1)

# Get true label index
true_label_idx = np.argmax(y_test[random_idx])
true_label = categories[true_label_idx]  # Convert index to category name

random_image_array = np.expand_dims(x_test[random_idx], axis=0)

# Predict class probabilities
prediction_probs = model.predict(random_image_array)
predicted_label_idx = np.argmax(prediction_probs)  # Get predicted label index
predicted_label = categories[predicted_label_idx]  # Convert index to category name

probabilities = {categories[i]: prediction_probs[0][i] * 100 for i in range(4)}

print("\n### Model Prediction on a Random Test Image ###")
print(f"True Label: {true_label}")
print(f"Predicted Label: {predicted_label}")
print(f"Prediction Probabilities: {probabilities}")
print("✅ Yay!" if predicted_label == true_label else "❌ Nay!")



