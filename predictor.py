# src/predictor.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from config import IMG_SIZE

class Predictor:
    def __init__(self, model_path):
        print("🔹 Loading model:", model_path)
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, img_path, class_names):
        # Load image
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_arr = image.img_to_array(img)

        # Preprocess theo EfficientNet
        img_arr = preprocess_input(img_arr)

        # Add batch dim
        img_arr = np.expand_dims(img_arr, axis=0)

        # Dự đoán
        preds = self.model.predict(img_arr)
        idx = np.argmax(preds)

        return {
            "label": class_names[idx],
            "prob": float(preds[0][idx])
        }
