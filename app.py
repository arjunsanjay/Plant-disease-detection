import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="🌿 Plant Disease Detection", layout="wide")

# Class labels
class_labels = [
    "Bacterial Leaf Blight", 
    "Brown Spot", 
    "Healthy", 
    "Leaf Blast", 
    "Leaf Scald", 
    "Narrow Brown Spot"
]

# Load models
model_paths = {
    "Custom CNN": "model/plant_disease_model.h5",
    "MobileNetV2": "model/mobilenet_model.h5",
    "MLP-Mixer": "model/mlp_mixer_model.h5"
}

models = {}
for name, path in model_paths.items():
    models[name] = tf.keras.models.load_model(path)

# Image preprocessing
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Sidebar
st.sidebar.image("assets/logo.png", width=150)
st.sidebar.title(" About")
st.sidebar.write("Upload an image of a plant leaf to detect its disease using 3 different models.")

# Main title
st.title("🌿 Plant Disease Detection System")
st.write("Upload an image below, and each model will predict the disease.")

# File uploader
uploaded_file = st.file_uploader("📸 Upload a Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save temporarily
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Predict with all models
    img_array = preprocess_image(temp_path)
    predictions = {}
    for name, model in models.items():
        pred = model.predict(img_array)
        predictions[name] = pred[0]

    # Display predictions
    st.subheader("🧠 Predictions from All Models:")
    for model_name, pred in predictions.items():
        pred_class = class_labels[np.argmax(pred)]
        st.write(f"**{model_name}**: 🌱 *{pred_class}*")

    # Plot probabilities
    st.subheader("📊 Prediction Probabilities Comparison")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(class_labels))
    width = 0.25

    for i, (model_name, pred) in enumerate(predictions.items()):
        ax.bar(x + i * width, pred, width, label=model_name)

    ax.set_ylabel('Probability')
    ax.set_title('Class Prediction Probabilities by Model')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.legend()
    st.pyplot(fig)

    os.remove(temp_path)
