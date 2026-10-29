import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 5

train_path = "train"
val_path = "val"
test_path = "test"

# ============================================================
# 2. DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    # Small rotations because knee X-rays should remain realistic
    rotation_range=8,

    # Small shifts
    width_shift_range=0.08,
    height_shift_range=0.08,

    # Small zoom
    zoom_range=0.10,

    # Slight shear
    shear_range=0.05,

    # Horizontal flip
    horizontal_flip=True,

    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# ============================================================
# 3. LOAD DATASET
# ============================================================

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_test_datagen.flow_from_directory(
    val_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_data = val_test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClass indices:")
print(train_data.class_indices)

print("\nTraining images:", train_data.samples)
print("Validation images:", val_data.samples)
print("Testing images:", test_data.samples)

# ============================================================
# 4. CALCULATE CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(
    enumerate(class_weights_array)
)

print("\nClass weights:")
for class_id, weight in class_weights.items():
    print(f"Class {class_id}: {weight:.4f}")

# ============================================================
# 5. LOAD MOBILENETV2
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze MobileNetV2
base_model.trainable = False

# ============================================================
# 6. BUILD MODEL
# ============================================================

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.BatchNormalization(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 8. MODEL SUMMARY
# ============================================================

print("\n========== MODEL SUMMARY ==========\n")
model.summary()

# ============================================================
# 9. CALLBACKS
# ============================================================

callbacks = [

    # Stop if validation loss doesn't improve
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    # Save the best model
    tf.keras.callbacks.ModelCheckpoint(
        "mobilenetv2_best.keras",
        monitor="val_accuracy",
        save_best_only=True
    ),

    # Reduce learning rate when validation loss stops improving
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.00001
    )
]

# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n========== TRAINING STARTED ==========\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ============================================================
# 11. TEST MODEL
# ============================================================

print("\n========== TESTING MODEL ==========\n")

test_loss, test_accuracy = model.evaluate(
    test_data
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ============================================================
# 12. SAVE FINAL MODEL
# ============================================================

model.save("mobilenetv2_oa_model.keras")

print("\nModel saved as:")
print("mobilenetv2_oa_model.keras")

# ============================================================
# 13. PLOT ACCURACY
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("MobileNetV2 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.show()

# ============================================================
# 14. PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("MobileNetV2 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.show()