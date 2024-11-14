import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# Suppress TensorFlow GPU warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Inject CSS to style the main block container, file uploader button, and other elements
custom_style = """
    <style>
        /* Set the main block container background color to white and text color to black */
        .stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 {
            background-color: white !important;
            color: black !important;
        }

        /* Set the main container (stMain) background color to white */
        .stMain.st-emotion-cache-bm2z3a.ea3mdgi8 {
            background-color: white !important;
        }

        /* Style the file uploader button with color #ae2740 */
        .st-emotion-cache-1erivf3.e1b2p2ww15 {
            background-color: #ae2740 !important;
            color: white !important;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }

        /* Style the file uploader button with the same color #ae2740 */
        .st-emotion-cache-15hul6a.ef3psqc16 {
            background-color: #ae2740 !important;
            color: white !important;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }

        /* Hide elements with the specified classes */
        .st-emotion-cache-7oyrr6.e1bju1570,
        .st-emotion-cache-1fttcpj.e1b2p2ww11,
        .eyeqlp53.st-emotion-cache-6rlrad.ex0cdmw0 { 
            display: none !important;
        }

        /* Hide the footer and toolbar */
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# Load precomputed feature vectors, filenames, and product IDs
feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
product_ids = pickle.load(open('product_ids.pkl', 'rb'))  # Load product IDs

# Initialize ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Ensure the 'Dataset' directory exists on your EC2 instance
dataset_dir = '/home/ubuntu/streamlitdemo/Dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Ensure the 'uploads' directory exists for saving uploaded images
upload_dir = 'uploads'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Save uploaded file to server
def save_uploaded_file(uploaded_file):
    try:
        upload_path = os.path.join(upload_dir, uploaded_file.name)
        with open(upload_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return upload_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Extract features from the uploaded image
def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

# Get recommendations based on feature similarity
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Generate WooCommerce product URL by product ID
def get_product_url(product_id):
    return f"https://cgbshop1.com/?p={product_id}"

# Normalize the path to the dataset folder
def get_normalized_path(filename):
    # Ensure filenames are correctly resolved to absolute paths under the Dataset directory
    return os.path.join(dataset_dir, filename)  # Absolute path to images in Dataset

# Ensure that filenames are correctly normalized
filenames = [get_normalized_path(f) for f in filenames]

# Check if a file exists at a given path
def open_recommended_image(image_path):
    print(f"Opening image at path: {image_path}")  # Debugging log to check the path
    if os.path.exists(image_path):
        try:
            recommended_image = Image.open(image_path)
            return recommended_image
        except FileNotFoundError:
            st.error(f"File not found: {image_path}")
    else:
        st.error(f"File does not exist: {image_path}")

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
        normalized_path = os.path.abspath(upload_path)  # Make sure the uploaded path is absolute
        features = extract_feature(normalized_path, model)
        
        # Get recommendations based on feature similarity
        indices = recommend(features, feature_list)

        # Get the number of recommended images
        num_recommendations = min(15, len(indices[0]))

        # Create columns dynamically based on the number of recommendations
        columns = st.columns(num_recommendations)

        for i in range(num_recommendations):
            with columns[i]:
                recommended_image_path = get_normalized_path(filenames[indices[0][i]])
                open_recommended_image(recommended_image_path)

                # Retrieve the product ID using the indices from product_ids
                product_id = product_ids[indices[0][i]]

                # Generate product URL
                product_url = get_product_url(product_id)

                # Display styled product link
                st.markdown(
                    f'<a href="{product_url}" style="color: #ae2740; text-decoration: none;">Voir les détails</a>',
                    unsafe_allow_html=True
                )

    else:
        st.header("Some error occurred in file upload")
