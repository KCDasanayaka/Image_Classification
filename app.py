# -*- coding: utf-8 -*-
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Set UTF-8 encoding for stdout to handle Unicode characters
sys.stdout.reconfigure(encoding='utf-8')

# Define the path to your model
model_path = 'd:/Projects/Image_Classification/my_model.h5'  # Update this path as needed

# Check if the model file exists
if os.path.exists(model_path):
    # Load the model
    model = load_model(model_path)
    print("Model loaded successfully.")
else:
    print(f"Model file not found at {model_path}. Please check the path.")
    exit(1)  # Exit the script if the model file is missing

# Function to load and preprocess the image
def load_and_prepare_image(image_path, target_size=(224, 224)):  # Adjust target size as needed
    img = load_img(image_path, target_size=target_size)
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)  # Make it batch-like
    img_arr = img_arr / 255.0  # Normalize if required by your model
    return img_arr

# Define the path to the image you want to classify
image_path = 'd:/Projects/Image_Classification/sample_image.jpg'  # Replace with the actual image path

# Check if the image file exists
if os.path.exists(image_path):
    # Load and prepare the image
    img_arr = load_and_prepare_image(image_path)
else:
    print(f"Image file not found at {image_path}. Please check the path.")
    exit(1)  # Exit the script if the image file is missing

# Perform prediction
try:
    prediction = model.predict(img_arr)
    
    # Process and print the prediction
    prediction_str = prediction.decode('utf-8') if isinstance(prediction, bytes) else str(prediction)
    print("Prediction:", prediction_str)

except UnicodeEncodeError as e:
    print("Unicode encoding error occurred:", e)

except Exception as e:
    print("An error occurred during prediction:", e)
