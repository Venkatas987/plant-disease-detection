import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# ---------------- PATHS ----------------
MODEL_PATH = os.path.join("models", "best_model.h5")
LABEL_PATH = os.path.join("models", "class_indices.json")

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# ---------------- LOAD CLASS LABELS ----------------
with open(LABEL_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index → class name
labels = {v: k for k, v in class_indices.items()}

# ---------------- PREDICTION FUNCTION ----------------
def predict(img):
    """
    Predict plant disease from a PIL Image.
    Returns: (label, confidence)
    """

    if not isinstance(img, Image.Image):
        img = Image.open(img)

    # Preprocess image
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions))
    class_index = int(np.argmax(predictions))
    label = labels[class_index]

    return label, confidence
