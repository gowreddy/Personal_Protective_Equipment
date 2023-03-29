#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow.keras as keras
from PIL import Image
import numpy as np


# In[ ]:


model = keras.models.load_model("modelppe.h5")

target_size = (128, 128)
st.set_page_config(
    page_title="Personal Protective Equipment",
    page_icon=":art:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# In[ ]:


uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])


# In[ ]:


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((128,128))
    image_array = np.array(image)
    image_array=np.expand_dims(image_array,axis=0)

    # Make a prediction
    predictions = model.predict(image_array)
    threshold=0.3

    # Get the predicted probabilities for each class
    probabilities = predictions[0]
    st.image(image, caption='uploaded_file', use_column_width=1000)
    # Check if the person is wearing a helmet
    st.write(predictions)
    st.write(probabilities)
    if (probabilities[0] and probabilities[2]) > threshold:
      st.write("The person is  wearing a helmet but not a mask.")
    elif (probabilities[0] and probabilities[3]) > threshold:
      st.write("The person is wearing a helmet but not a mask.")
    elif (probabilities[1] and probabilities[3]) > threshold:
      st.write("The person is neither wearing a mask and a helmet.")
    else:
      st.write("The person is wearing a mask but not a helmet.")

