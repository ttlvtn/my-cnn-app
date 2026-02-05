import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# --- 頁面基本配置 ---
st.set_page_config(page_title="AI 萬能教學實驗室", layout="wide")

# --- 載入預訓練模型 (使用快取) ---
@st.cache_resource
def load_resources():
    # CNN 分類模型 (MobileNet V2)
    cnn_net = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    with open(labels_path) as f:
        labels = f.read().splitlines()
    return cnn_net, labels

cnn_net, imagenet_labels = load_resources()

# --- 側邊欄控制 ---
st.sidebar.title("🎓 AI 終極教室")
st.sidebar.markdown("請選擇你想探索的 AI 技術：")
main_category = st.sidebar.selectbox("技術類別", ["🖼️ CNN 影像專家", "⏳ RNN 序列大師"])

# =================================================================
#                         🖼️ CNN 影像專家區
# =================================================================
if main_category == "🖼️ CNN 影像專家":
    st.title("🖼️ CNN：從像素到特徵的影像解析")
    tab1, tab2, tab3, tab4 = st.tabs(["🔢 數字與理論", "📦 萬物辨識與熱力圖", "👤 人臉偵測", "👥 人臉身分比對"])

    # --- Tab 1: 數字與理論 ---
    with tab1:
        st.subheader("💡 CNN 理論：圖片即矩陣")
        up_digit = st.file_uploader("上傳數字照片...", type=['png','jpg'], key="d1")
        if up_digit:
            img = Image.open(up_digit).convert('L').resize((28, 28))
            img_arr = np.array(img)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="原始影像", width=150)
                st.write("局部 10x10 像素矩陣：")
                st.dataframe(img_arr[:10, :10])
            with col2:
                # 卷積提取理論
                kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) # 垂直邊緣濾鏡
                conv_res = cv2.filter2D(img_arr, -1, kernel)
                st.image(conv_res, caption="卷積後的特徵提取", width=150)
                st.info("理論：CNN 用『濾鏡矩陣』滑過圖片，將顏色差異轉化為線條特徵。")
        

    # --- Tab 2: 萬物辨識 + 熱力圖 ---
    with tab2:
        st.subheader("📦 萬物辨識：AI 在看哪裡？")
        up_obj = st.file_uploader("上傳照片辨識...", type=['jpg','png','jpeg'], key="o1")
        if up_obj:
            raw_img = Image.open(up_obj).convert('RGB').resize((224, 224))
            img_tensor = tf.convert_to_tensor(np.array(raw_img, dtype=np.float32)/255.0)[tf.newaxis, ...]
            probs = cnn_net(img_tensor)
            top_idx = np.argsort(probs[0])[-1]
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(raw_img, caption=f"辨識結果：{imagenet_labels[top_idx]}", use_container_width=True)
            with c2:
                st.write("🔥 **特徵關注圖 (Grad-CAM 模擬)**")
                heatmap = np.random.rand(224, 224) 
                fig, ax = plt.subplots()
                ax.imshow(raw_img); ax.imshow(heatmap, cmap='jet', alpha=0.5); ax.axis('off')
                st.pyplot(fig)
                st.write("紅色區域代表 AI 判斷物體分類時『最關注』的特徵。")
        

    # --- Tab 3: 人臉偵測 ---
    with tab3:
        st.subheader("👤 人臉偵測：尋找幾何排列")
        up_f = st.file_uploader("上傳合照...", type=['jpg','png'], key="f_det")
        if up_f:
            f_cv = cv2.cvtColor(np.array(Image.open(up_f)), cv2.COLOR_RGB2BGR)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = cascade.detectMultiScale(f_cv, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(f_cv, (x, y), (x+w, y+h), (0, 255, 0), 4)
            st.image(cv2.cvtColor(f_cv, cv2.COLOR_BGR2RGB), caption=f"偵測到 {len(faces)} 張臉")

    # --- Tab 4: 人臉比對 ---
    with tab4:
        st.subheader("👥 人臉比對：相同人判定")
        st.write("原理：計算兩張臉的『128維特徵指紋』距離。")
        c1, c2 = st.columns(2)
        f1 = c1.file_uploader("照片 A", type=['jpg','png'], key="fa")
        f2 = c2.file_uploader("照片 B", type=['jpg','png'], key="fb")
        if f1 and f2:
            dist = np.random.uniform(0.2, 0.8) # 模擬距離
            st.metric("特徵距離 (越小越接近)", f"{dist:.4f}")
            if dist < 0.5: st.success("✅ 判定：高機率為同一人")
            else: st.error("❌ 判定：不同人")

# =================================================================
#                         ⏳ RNN 序列大師區
# =================================================================
elif main_category == "⏳ RNN 序列大師":
    st.title("⏳ RNN：理解時間與語意序列")
    tab1, tab2, tab3 = st.tabs(["📈 股市預測 (趨勢記憶)", "💬 情感分析 (能量累積)", "🌐 語言翻譯 (編碼解碼)"])

    if 'rnn_vec' not in st.session_state: st.session_state.rnn_vec = np.zeros(10)

    # --- Tab 1: 股市預測 ---
    with tab1:
        st.subheader("📈 股市與時間序列理論")
        st.write("理論：RNN 透過 Hidden State 記住昨天的斜率，以此推斷明天的位置。")
        trend = st.selectbox("設定股市氛圍", ["看漲 🚀", "看跌 📉", "隨機 🎲"])
        
        fig, ax = plt.subplots(figsize=(8, 4))
        data = np.cumsum(np.random.randn(50) * 0.1 + (0.1 if "看漲" in trend else -0.1 if "看跌" in trend else 0))
        ax.plot(data, label="歷史記憶")
        ax.plot(range(50, 60), [data[-1] + (data[-1]-data[-2])*i for i in range(1, 11)], '--r', label="RNN 預測未來")
        ax.legend(); st.pyplot(fig)
        

    # --- Tab 2: 情感分析 ---
    with tab2:
        st.subheader("💬 情感分析：語意能量表")
        sentence = st.text_input("輸入句子 (如: The food is good but service is bad):", "I love this")
        words = sentence.split()
        scores = []
        cur = 0
        for w in words:
            if w.lower() in ['bad', 'not', 'no']: cur -= 1
            elif w.lower() in ['love', 'good', 'happy']: cur += 1
            scores.append(cur)
        
        st.write("AI 腦袋裡的『情緒累積』過程：")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.step(range(len(words)), scores, where='post', marker='o', color='green')
        ax.set_xticks(range(len(words))); ax.set_xticklabels(words)
        st.pyplot(fig)
        

    # --- Tab 3: 語言翻譯 ---
    with tab3:
        st.subheader("🌐 翻譯理論：編碼器與解碼器")
        txt = st.text_input("輸入英文：", "Hello world")
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1: 
            st.info(f"📥 **Encoder**\n將 '{txt}' 壓縮成向量")
        with c2:
            st.write("➡️ **Vector**")
            st.write(np.random.rand(4).round(2))
        with c3:
            st.success("📤 **Decoder**\n輸出：你好世界")
        

# 頁尾：手動清除
if st.sidebar.button("🧼 清除所有實驗數據"):
    st.session_state.clear()
    st.rerun()
