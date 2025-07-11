# trained_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# ---------------------- Config ----------------------
DATASET_DIR = "crop_image"  # Folder must have subfolders like rice/, wheat/, etc.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_NAME = "keras_model.h5"
LABELS_FILE = "labels.txt"

# ---------------------- Data Generator ----------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ---------------------- Model Definition ----------------------
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------- Train Model ----------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# ---------------------- Save Model & Labels ----------------------
model.save("plant_model.h5")
with open(LABELS_FILE, "w") as f:
    f.write("\n".join(train_data.class_indices.keys()))

print("✅ Model trained and saved as plant_predict.h5")
