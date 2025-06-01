import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']

def load_models():
    cnn = load_model("Models/model.keras")
    mobilenet = load_model("Models/mobilenet_model.keras")
    resnet = load_model("Models/resnet_model.keras")
    return cnn, mobilenet, resnet

def predict_all_models(image_array, cnn, mobilenet, resnet):
    models = {
        "CNN": cnn,
        "MobileNetV2": mobilenet,
        "ResNet50": resnet
    }

    predictions = {}
    for name, model in models.items():
        probs = model.predict(image_array)[0]
        class_idx = np.argmax(probs)
        predictions[name] = (CLASS_NAMES[class_idx], probs[class_idx])
    
    return predictions


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
