import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.preprocessing import image


model = tf.keras.models.load_model('trained_model.h5')


st.markdown("<h1 style='text-align: center;'># Cat or Dog Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an image to predict whether it's a cat or a dog.")


def predict_image(image):
    image = np.array(image)
    image = tf.image.resize(image, (64, 64))  
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  

    prediction = model.predict(image)
    if prediction[0][0] > 0.5:
        predicted_class = 'Dog'
        background_color = 'red'  
    else:
        predicted_class = 'Cat'
        background_color = 'green'  

    return predicted_class, background_color


uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    predicted_class, background_color = predict_image(Image.open(uploaded_image))
    
    st.markdown(
        f"<div style='background-color: {background_color}; padding: 10px; text-align: center;'>"
        f"<h3>The image is more likely to be a '{predicted_class}'.</h3>"
        "</div>",
        unsafe_allow_html=True
    )