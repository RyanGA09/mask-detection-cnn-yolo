
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Load data dari folder kaggle
base_dir = os.path.join('data', 'Face Mask Dataset', 'Face Mask Dataset')

train_path = os.path.join(base_dir, 'Train')
val_path = os.path.join(base_dir, 'Validation')
test_path = os.path.join(base_dir, 'Test')

# Preprocessing data
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_path, target_size=(150, 150), batch_size=32, class_mode='binary')
val_data = datagen.flow_from_directory(val_path, target_size=(150, 150), batch_size=32, class_mode='binary')
test_data = datagen.flow_from_directory(test_path, target_size=(150, 150), batch_size=32, class_mode='binary')

# Model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('models/mask_model.h5', save_best_only=True)
model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[checkpoint])

model.evaluate(test_data)
