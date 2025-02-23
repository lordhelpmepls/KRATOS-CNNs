 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:30:05 2025

@author: marlinmahmud
"""


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



# For few images 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation for Training Data
datagen = ImageDataGenerator(
    rotation_range=30,  # Rotate images up to 30 degrees
    width_shift_range=0.2,  # Shift image horizontally
    height_shift_range=0.2,  # Shift image vertically
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Flip horizontally
    fill_mode='nearest'  # Fill missing pixels
)

# Fit augmentation generator to training data
datagen.fit(x_train)





# Build CNN Model
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Four output classes
])



# For few images

from tensorflow.keras.optimizers import Adam

# Lower learning rate
optimizer = Adam(learning_rate=0.0001)  

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with augmented data 
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=8),  
    epochs=50,  # Increase epochs since augmentation helps prevent overfitting
    validation_data=(x_val, y_val),
    steps_per_epoch=len(x_train) // 8,  
    verbose=1
)




'''
# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Multi-class classification
              metrics=['accuracy'])
'''





# Train 
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stop]
)

# Evaluate 
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")



# Training and validation loss
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

model.save("galaxy_morphology_classification_K3.keras")




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
    




# Select a random image from test set
random_idx = random.randint(0, len(x_test) - 1)

# Get the true label index
true_label_idx = np.argmax(y_test[random_idx])  
true_label = categories[true_label_idx]  # Convert index to category name

random_image_array = np.expand_dims(x_test[random_idx], axis=0)

# Predict the class probabilities
prediction_probs = model.predict(random_image_array)
predicted_label_idx = np.argmax(prediction_probs)  # Get predicted label index
predicted_label = categories[predicted_label_idx]  # Convert index to category name

probabilities = {categories[i]: prediction_probs[0][i] * 100 for i in range(4)}

print("\n### Model Prediction on a Random Test Image ###")
print(f"True Label: {true_label}")
print(f"Predicted Label: {predicted_label}")
print(f"Prediction Probabilities: {probabilities}")
print("✅ Yay!" if predicted_label == true_label else "❌ Nay!")


    
    
    
    
    
    