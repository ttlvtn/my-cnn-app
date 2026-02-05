import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 頁面基本設定
st.set_page_config(page_title="AI 偵探實驗室", layout="wide")

# --- 模擬模型區 (為了教學方便，我們使用內建權重或簡單結構) ---
@st.cache_resource
def get_models():
    # 建立一個簡單的 CNN 模型用於 MNIST (28x28 灰階)
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1), name='conv_1'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # 建立一個簡單的 RNN 模型 (用於情感分析示範)
    rnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(1000, 16, input_length=10),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return cnn_model, rnn_model

cnn_model, rnn_model = get_models()

# --- 側邊欄：教學導航 ---
st.sidebar.title("🧠 AI 教室選單")
mode = st.sidebar.radio("請選擇學習主題：", ["1. CNN 影像辨識 (超級掃描眼)", "2. RNN 序列預測 (記憶筆記本)"])

# --- 主畫面：CNN 教學 ---
if mode == "1. CNN 影像辨識 (超級掃描眼)":
    st.title("🖼️ CNN：AI 如何看懂圖片？")
    st.write("CNN 就像偵探，會用**濾鏡(Filter)**去掃描圖片的每一個角落，尋找特定的線條或形狀。")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("第一步：提供線索")
        uploaded_file = st.file_uploader("上傳一張手寫數字圖片 (0-9)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert('L').resize((28, 28))
            st.image(img, caption="AI 看到的原始圖片", width=150)
            
            # 預處理
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # 預測
            pred = cnn_model.predict(img_array)
            st.metric("AI 猜測這個數字是", np.argmax(pred))
            st.write(f"信心程度：{np.max(pred)*100:.2f}%")

    with col2:
        st.subheader("第二步：AI 偵探視角 (視覺化)")
        if uploaded_file:
            # 視覺化第一層卷積
            # 建立一個只輸出第一層結果的模型
            layer_output = cnn_model.get_layer('conv_1').output
            vis_model = tf.keras.models.Model(inputs=cnn_model.input, outputs=layer_output)
            features = vis_model.predict(img_array)
            
            fig, axes = plt.subplots(2, 2)
            for i in range(4):
                ax = axes[i//2, i%2]
                ax.imshow(features[0, :, :, i], cmap='viridis')
                ax.axis('off')
                ax.set_title(f"濾鏡 {i+1} 抓到的特徵")
            st.pyplot(fig)
            st.info("💡 看到上面的彩色圖了嗎？這就是 AI 在尋找邊緣和轉角！不同的濾鏡會抓到不同的線索。")

# --- 主畫面：RNN 教學 ---
elif mode == "2. RNN 序列預測 (記憶筆記本)":
    st.title("⏳ RNN：AI 如何理解順序？")
    st.write("RNN 不只看現在，還會**記得過去**。這對理解句子或預測股價非常重要。")
    
    st.subheader("實驗：文字的順序重要嗎？")
    text_input = st.text_input("輸入一句話 (英文)，讓 AI 判斷情感：", "I am very happy today")
    
    # 模擬 RNN 的記憶過程
    words = text_input.split()
    st.write("### 🧠 AI 的記憶流：")
    
    memory_display = []
    for word in words:
        memory_display.append(f"[{word}]")
        st.write(" ➔ ".join(memory_display) + " (寫入筆記本...)")
        
    st.write("---")
    # 隨機模擬一個預測結果 (實際教學可用更複雜的模型)
    sentiment_score = np.random.random()
    if sentiment_score > 0.5:
        st.success(f"😊 情感分析結果：正面 (分數: {sentiment_score:.2f})")
    else:
        st.error(f"😢 情感分析結果：負面 (分數: {sentiment_score:.2f})")
    
    st.info("💡 教學重點：RNN 會一個字一個字讀，並把之前的訊息傳給下一個狀態。如果把順序打亂，AI 讀到的意思就會完全不同！")

# --- 頁尾教學總結 ---
st.markdown("---")
st.caption("🚀 這是為國高中生設計的 AI 教學工具。CNN 負責空間（看圖），RNN 負責時間（記順序）。")
