import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "final_model", "best_efficientnet_model.keras")

model = load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
    safe_mode=False
)

def prepare_model_input(img_bgr):
    img_resized = cv2.resize(img_bgr, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.array(img_rgb, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_leaf_class(img_bgr):
    img_array = prepare_model_input(img_bgr)
    preds = model.predict(img_array, verbose=0)[0]

    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    pred_class = CLASS_NAMES[pred_idx]

    return {
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "all_probabilities": preds.tolist(),
        "model": model,
        "model_input": img_array,
        "class_names": CLASS_NAMES
    }