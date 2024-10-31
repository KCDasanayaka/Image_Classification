import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os

# Set header
st.header('Image Classification Model')

# Load model
model = load_model('D:\\Projects\\Image_Classification\\Image_classify.keras')

# Data categories
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 
    'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 
    'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Set image size
img_height, img_width = 180, 180

# Load image
image_path = st.text_input('Enter the path of the image:', 'D:\\Projects\\Image_Classification\\Apple.jpg')

# Check if file exists and is valid
if os.path.exists(image_path):
    try:
        image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image_load)
        img_bat = tf.expand_dims(img_arr, 0)

        # Predict
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict[0])

        # Display results
        st.image(image_path, width=200)
        st.write('Predicted Veg/Fruit: ' + data_cat[np.argmax(score)])
        st.write('Confidence: {:.2f}%'.format(np.max(score) * 100))
        
    except Exception as e:
        st.error(f"Error loading the image. Please check the image path and format. Error: {e}")
else:
    st.warning("The specified image path does not exist. Please enter a valid path.")
