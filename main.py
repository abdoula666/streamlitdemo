import os
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Disable GPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Path normalization function for cross-platform compatibility
def normalize_path(path):
    return os.path.normpath(path)

# Load precomputed feature vectors, filenames, and product IDs
feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
product_ids = pickle.load(open('product_ids.pkl', 'rb'))

# Initialize ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Save uploaded file to the server
def save_uploaded_file(uploaded_file):
    try:
        upload_path = os.path.join('uploads', uploaded_file.name)
        with open(upload_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return upload_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Extract features from the uploaded image
def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    if img is None:
        st.error(f"Failed to load image from path: {img_path}")
        return None
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

# Recommend based on feature similarity
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# WooCommerce product URL generator
def get_product_url(product_id):
    return f"https://cgbshop1.com/?p={product_id}"

# Normalize all filenames
filenames = [normalize_path(f) for f in filenames]

# Main Streamlit app code
uploaded_file = st.file_uploader("Choisir l'image")  # Update label to French
if uploaded_file is not None:
    upload_path = save_uploaded_file(uploaded_file)
    
    if upload_path:
        # Display the uploaded image
        display_image = Image.open(upload_path)
        resized_img = display_image.resize((200, 200))
        st.image(resized_img)
        
        # Extract features from the uploaded image
        normalized_path = normalize_path(upload_path)
        features = extract_feature(normalized_path, model)
        
        if features is not None:
            # Get recommendations based on feature similarity
            indices = recommend(features, feature_list)

            # Get the number of recommended images
            num_recommendations = min(15, len(indices[0]))

            # Create columns dynamically based on the number of recommendations
            columns = st.columns(num_recommendations)

            for i in range(num_recommendations):
                with columns[i]:
                    # Get the normalized path for the recommended image
                    recommended_image_path = normalize_path(filenames[indices[0][i]])

                    try:
                        # Try opening the image using the normalized path
                        recommended_image = Image.open(recommended_image_path)
                        st.image(recommended_image)
                    except FileNotFoundError:
                        st.error(f"File not found: {recommended_image_path}")

                    # Retrieve the product ID using the indices from product_ids
                    product_id = product_ids[indices[0][i]]

                    # Generate product URL
                    product_url = get_product_url(product_id)

                    # Display the product link
                    st.markdown(
                        f'<a href="{product_url}" style="color: #ae2740; text-decoration: none;">Voir les détails</a>',
                        unsafe_allow_html=True
                    )

    else:
        st.header("Some error occurred in file upload")
