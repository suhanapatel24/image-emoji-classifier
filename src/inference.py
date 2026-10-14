from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = Path("emoji_classifier_model.h5")
LABELS_PATH = Path("label_classes.npy")
IMAGE_SIZE = (64, 64)

def load_assets(model_path=MODEL_PATH, labels_path=LABELS_PATH):
    model = tf.keras.models.load_model(model_path, compile=False)
    labels = np.load(labels_path, allow_pickle=True)
    return model, labels

def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)

def predict_top_k(model, labels, image: Image.Image, k=5):
    probabilities = model.predict(preprocess_image(image), verbose=0)[0]
    k = min(k, len(probabilities))
    indices = np.argsort(probabilities)[-k:][::-1]
    return [
        {"label": str(labels[index]), "confidence": float(probabilities[index])}
        for index in indices
    ]
