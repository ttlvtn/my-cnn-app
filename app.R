import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("我的第一個 CNN 應用")

# 建立一個簡單的模型結構（實際應用建議上傳訓練好的 .h5 檔）
def load_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

model = load_model()

uploaded_file = st.file_uploader("請上傳一張圖片...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption='上傳的圖片', use_container_width=True)
    
    # 簡單預測邏輯
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    
    st.write(f"預測數字為: {np.argmax(prediction)}")
    st.bar_chart(prediction[0])
