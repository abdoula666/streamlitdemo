import streamlit as st
import os
import numpy as np
import pickle
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Inject CSS for styling
st.markdown("""
    <style>
        .stMainBlockContainer, .stMain {background-color: white !important;}
        .st-emotion-cache-1erivf3, .st-emotion-cache-15hul6a {
            background-color: #ae2740 !important; color: white !important; 
            padding: 10px 20px; border: none; border-radius: 4px; 
            cursor: pointer; font-weight: bold;}
        .st-emotion-cache-7oyrr6, .st-emotion-cache-1fttcpj,
        .eyeqlp53.st-emotion-cache-6rlrad {display: none !important;}
        footer, #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load data
features = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
product_ids = pickle.load(open('product_ids.pkl', 'rb'))

# Initialize ResNet50 model
model = tf.keras.Sequential([
    ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    GlobalMaxPooling2D()
])
model.trainable = False

# Ensure upload directory exists
upload_dir = '/home/ubuntu/streamlitdemo/uploads'
os.makedirs(upload_dir, exist_ok=True)

# Save uploaded file
def save_file(uploaded_file):
    file_path = os.path.join(upload_dir, uploaded_file.name.replace("\\", "/"))
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Extract features
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (224, 224))
    pre_img = preprocess_input(np.expand_dims(img, axis=0))
    return model.predict(pre_img).flatten() / norm(result)

# Recommend similar products
def recommend(features):
    neighbors = NearestNeighbors(n_neighbors=15, algorithm='brute')
    neighbors.fit(features)
    return neighbors.kneighbors([features])[1]

# Generate product URL
def product_url(product_id):
    return f"https://cgbshop1.com/?p={product_id}"

# Main app logic
uploaded_file = st.file_uploader("Choisir l'image")
if uploaded_file:
    file_path = save_file(uploaded_file)
    if file_path:
        st.write(f"File saved to: {file_path}")
        st.image(Image.open(file_path).resize((200, 200)))
        features = extract_features(file_path)
        if features is not None:
            indices = recommend(features)
            for i in range(min(15, len(indices[0]))):
                col = st.columns(15)[i]
                with col:
                    img_path = filenames[indices[0][i]]
                    st.image(Image.open(img_path))
                    st.markdown(f'<a href="{product_url(product_ids[indices[0][i]])}" style="color: #ae2740;">Voir les détails</a>', unsafe_allow_html=True)
