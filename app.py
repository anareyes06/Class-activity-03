import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from utils import preprocess_image, load_models, predict_all_models


st.set_page_config(page_title="Bean Leaf Classifier", layout="centered")

st.title("Bean Leaf Disease Classifier 🌱")

# Subir imagen
uploaded_file = st.file_uploader("Upload a bean leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocesar imagen
    img_array = preprocess_image(uploaded_file)

    # Cargar modelos
    cnn_model, mobilenet_model, resnet_model = load_models()

    # Obtener predicciones
    predictions = predict_all_models(img_array, cnn_model, mobilenet_model, resnet_model)

    # Mostrar resultados
    st.subheader("Predictions")
    for model_name, (label, conf) in predictions.items():
        st.write(f"**{model_name}**: {label} ({conf:.2f} confidence)")

    # Métricas (pueden ser estáticas si las calculaste antes)
    st.subheader("Classification Metrics")
    st.markdown("""
    - **Accuracy**:
        - CNN: 94.2%
        - MobileNetV2: 97.8%
        - ResNet50: 98.5%
    - **F1-score**: 
        - CNN: 0.941
        - MobileNetV2: 0.975
        - ResNet50: 0.984
    """)