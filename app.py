import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 設定頁面
st.set_page_config(page_title="AI 萬能實驗室", layout="wide")

# --- 載入預訓練模型 (MobileNet 用於 CNN, 簡單 RNN 模擬) ---
@st.cache_resource
def load_models():
    # 載入 Google 的 MobileNet (可辨識 1000 種物體)
    cnn_model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")
    # 讀取標籤檔 (ImageNet)
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    with open(labels_path) as f:
        labels = f.read().splitlines()
    return cnn_model, labels

cnn_net, imagenet_labels = load_models()

# --- 側邊欄設計 ---
st.sidebar.title("🧪 AI 即時實驗室")
st.sidebar.markdown("這是一個可以讓你「隨便測試」的 AI 教室。")
mode = st.sidebar.selectbox("切換技術", ["CNN 影像專家 (萬物辨識)", "RNN 序列大師 (對話與記憶)"])

# ================= CNN 影像專家 =================
if mode == "CNN 影像專家 (萬物辨識)":
    st.title("🖼️ CNN：只要是圖片，我都認得！")
    st.write("這個模型學習過 1000 種物體，你可以上傳任何照片試試看。")
    
    img_file = st.file_uploader("📸 上傳照片 (貓、狗、車、水果等...)", type=['jpg', 'png', 'jpeg'])
    
    if img_file:
        img = Image.open(img_file).convert('RGB').resize((224, 224))
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img, caption="AI 正在觀察這張圖...", use_container_width=True)
            # 預處理
            img_arr = np.array(img) / 255.0
            img_arr = img_arr[np.newaxis, ...]
            
            # 推論
            probs = cnn_net(img_arr)
            top_3_indices = np.argsort(probs[0])[-3:][::-1]
            
        with col2:
            st.subheader("📊 辨識結果與信心度")
            for i in top_3_indices:
                label = imagenet_labels[i]
                score = float(probs[0][i])
                st.write(f"**{label.capitalize()}**")
                st.progress(min(max(score, 0.0), 1.0))
            
            st.info("💡 **教學點**：看到上面的機率分布了嗎？CNN 並不是『絕對肯定』，它是在做機率判斷！")

# ================= RNN 序列大師 =================
elif mode == "RNN 序列大師 (對話與記憶)":
    st.title("⏳ RNN：給我文字，我給你記憶！")
    st.write("輸入任何句子，觀察 AI 如何在腦中累積記憶數值。")

    if 'rnn_mem' not in st.session_state:
        st.session_state.rnn_mem = []
        st.session_state.rnn_vec = np.zeros(10)

    # 模擬股市、翻譯與對話的綜合體驗
    input_text = st.text_input("💬 跟 AI 說句話或是打個股價趨勢 (例如: Happy, Down, buy):")
    
    if st.button("送入記憶鏈"):
        if input_text:
            st.session_state.rnn_mem.append(input_text)
            # 模擬 RNN 數值跳動
            change = (np.random.rand(10) - 0.5) * 0.5
            st.session_state.rnn_vec = np.clip(st.session_state.rnn_vec + change, -1, 1)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔗 記憶序列")
        if st.session_state.rnn_mem:
            st.code(" ➔ ".join(st.session_state.rnn_mem))
            if st.button("🧼 歸零記憶"):
                st.session_state.rnn_mem = []
                st.session_state.rnn_vec = np.zeros(10)
                st.rerun()
        else:
            st.write("目前是一張白紙...")

    with col2:
        st.subheader("🔢 隱藏狀態 (Hidden State) 能量圖")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#FF4B4B' if x < 0 else '#00CC96' for x in st.session_state.rnn_vec]
        ax.bar(range(10), st.session_state.rnn_vec, color=colors)
        ax.set_ylim(-1.2, 1.2)
        st.pyplot(fig)

    st.write("---")
    st.subheader("🔮 RNN 的多重應用預測")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.write("**📈 股市/趨勢**")
        st.line_chart(np.cumsum(st.session_state.rnn_vec))
    with c2:
        st.write("**🌐 翻譯意圖**")
        st.write("AI 捕捉到的語意權重：" + str(np.abs(st.session_state.rnn_vec).mean().round(2)))
    with c3:
        st.write("**💬 對話情緒**")
        sentiment = "正面" if st.session_state.rnn_vec.sum() > 0 else "負面"
        st.write(f"目前情緒判定：{sentiment}")
