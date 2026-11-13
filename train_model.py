# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# 2. DEFINE PARAMETERS
# ==========================================

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10

# Dataset classes
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ==========================================
# 3. FIND DATASET LOCATION
# ==========================================

possible_paths = [
    Path("chest_xray"),
    Path("chest_xray/chest_xray"),
    Path("dataset/chest_xray"),
    Path("dataset/chest_xray/chest_xray")
]

DATA_DIR = None

for path in possible_paths:

    if (path / "train").exists():

        DATA_DIR = path
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Dataset not found. Check your chest_xray folder."
    )

print("Dataset location:", DATA_DIR.resolve())


# ==========================================
# 4. LOAD TRAINING DATA
# ==========================================

train_data = tf.keras.utils.image_dataset_from_directory(

    DATA_DIR / "train",

    image_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    color_mode="grayscale",

    label_mode="binary",

    class_names=CLASS_NAMES,

    shuffle=True

)


# ==========================================
# 5. LOAD VALIDATION DATA
# ==========================================

val_data = tf.keras.utils.image_dataset_from_directory(

    DATA_DIR / "val",

    image_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    color_mode="grayscale",

    label_mode="binary",

    class_names=CLASS_NAMES,

    shuffle=False

)


# ==========================================
# 6. LOAD TESTING DATA
# ==========================================

test_data = tf.keras.utils.image_dataset_from_directory(

    DATA_DIR / "test",

    image_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    color_mode="grayscale",

    label_mode="binary",

    class_names=CLASS_NAMES,

    shuffle=False

)


print("Classes:", train_data.class_names)


# ==========================================
# 7. BUILD ANN ARCHITECTURE
# ==========================================

model = models.Sequential([

    # Input layer
    layers.Input(shape=(128, 128, 1)),

    # Normalize pixel values between 0 and 1
    layers.Rescaling(1./255),

    # Convert image into one-dimensional array
    layers.Flatten(),

    # First hidden layer
    layers.Dense(128, activation="relu"),

    # Prevent overfitting
    layers.Dropout(0.3),

    # Second hidden layer
    layers.Dense(64, activation="relu"),

    # Output layer
    layers.Dense(1, activation="sigmoid")

])


# ==========================================
# 8. COMPILE ANN MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(name="auc")
    ]

)


# ==========================================
# 9. DISPLAY MODEL ARCHITECTURE
# ==========================================

model.summary()


# ==========================================
# 10. EARLY STOPPING
# ==========================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)


# ==========================================
# 11. TRAIN ANN MODEL
# ==========================================

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    callbacks=[early_stop]

)


# ==========================================
# 12. EVALUATE MODEL
# ==========================================

results = model.evaluate(test_data)

print("\nTEST RESULTS")

for name, value in zip(model.metrics_names, results):

    print(name, ":", value)


# ==========================================
# 13. SAVE TRAINED MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

model.save("model/ann_chest_xray.keras")

print("Model saved successfully!")


# ==========================================
# 14. VISUALIZE TRAINING ACCURACY
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("ANN Model Accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig("model/accuracy.png")

plt.show()


# ==========================================
# 15. VISUALIZE TRAINING LOSS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("ANN Model Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.savefig("model/loss.png")

plt.show()

