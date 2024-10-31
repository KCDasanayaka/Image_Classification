import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Header for the Streamlit app
st.header('Image Classification Model')

# Load the pre-trained model
model = load_model('D:/Projects/Image_Classification/Image_classify.keras')

# List of categories for classification
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Set the target image size
img_height = 180
img_width = 180

# Input for image path
image_path = st.text_input('Enter Image Path', 'Apple.jpg')

try:
    # Load and preprocess the image
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(image_load)
    img_batch = tf.expand_dims(img_array, 0)  # Create a batch dimension

    # Display the input image
    st.image(image_path, width=200)

    # Make a prediction
    predictions = model.predict(img_batch)
    score = tf.nn.softmax(predictions[0])

    # Display the prediction results
    st.write('Prediction: This is likely a(n) **{}**.'.format(data_cat[np.argmax(score)]))
    st.write('Confidence: {:.2f}%'.format(100 * np.max(score)))

except Exception as e:
    st.error("Error loading the image. Please check the image path and format.")
    st.write(e)
