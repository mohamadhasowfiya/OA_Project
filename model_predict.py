import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array


# ============================================================
# 1. MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR.parent / "mobilenetv2_oa_model.keras"


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("Loading OA MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("OA MobileNetV2 model loaded successfully!")


# ============================================================
# 3. OA CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Healthy",
    1: "Doubtful",
    2: "Minimal",
    3: "Moderate",
    4: "Severe"
}


# ============================================================
# 4. PREDICT X-RAY
# ============================================================

def predict_xray(image_path):

    # Load image
    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    # Convert image to NumPy array
    image_array = img_to_array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get highest probability class
    predicted_class = int(
        np.argmax(predictions)
    )

    # Get confidence
    confidence = float(
        predictions[predicted_class]
    )

    # Return result
    return {
        "class_id": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2)
    }