# pip install streamlit
# pip install streamlit-drawable-canvas
import streamlit as st 
from streamlit_drawable_canvas import st_canvas
from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray, rgba2rgb

import numpy as np  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json

# 模型載入
model = tf.keras.models.load_model('model.h5')
st.title('Crab_英文字母大小寫及數字辨識')
col1, col2 = st.columns(2)

with col1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=10,
        stroke_color="rgba(0, 0, 0, 1)",
        update_streamlit=True,
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas1",
    )

with col2:
    if st.button('辨識'):
        # print(canvas_result.image_data.shape)
        image1 = rgb2gray(rgba2rgb(canvas_result.image_data))
        image_resized = resize(image1, (28, 28), anti_aliasing=True)  
        # print(image_resized)
        X1 = image_resized.reshape(1,28,28) # / 255
        X1 = np.abs(1-X1)
        # class_names_1 = [chr(ord('0')+i) for i in range(10)]
        # class_names_2 = [chr(ord('A')+i) for i in range(26)]
        # class_names_3 = [chr(ord('a')+i) for i in range(26)]
        st.write("predict...")
        predictions = np.argmax(model.predict(X1), axis=-1)
        st.write(predictions[0])
        if predictions[0] <10:
            st.write('# ' + chr(ord('0')+ predictions[0]))
        elif predictions[0] <36:
            st.write('# ' + chr(ord('A')+ predictions[0])-10)
        else:
            st.write('# ' + chr(ord('a')+ predictions[0])-36)
        st.image(image_resized)
