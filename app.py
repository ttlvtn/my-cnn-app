import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. 頁面配置
st.set_page_config(page_title="AI 偵探教室：CNN vs RNN", layout="wide")

# 2. 核心模型載入 (使用快取避免重複載入)
@st.cache_resource
def load_teaching_models():
    # 建立一個簡單的 CNN 模型
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1), name='conv_layer'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return cnn

cnn_model = load_teaching_models()

# 3. 側邊欄導覽
st.sidebar.title("🎓 AI 教學實驗室")
st.sidebar.info("這是一個專為學生設計的 AI 視覺化工具。")
teaching_mode = st.sidebar.radio("切換教學主題：", ["🖼️ CNN 影像掃描眼", "⏳ RNN 序列記憶力"])

# --- 模式 1：CNN 影像辨識 ---
if teaching_mode == "🖼️ CNN 影像掃描眼":
    st.title("🖼️ CNN (卷積神經網絡)")
    st.write("### 教學目標：理解 AI 如何透過「濾鏡」觀察圖片特徵。")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🕵️ 觀察：上傳一張手寫數字")
        uploaded_file = st.file_uploader("選擇 JPG/PNG 圖片...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            # 圖片預處理
            raw_img = Image.open(uploaded_file).convert('L').resize((28, 28))
            st.image(raw_img, caption="AI 看到的輸入圖", width=200)
            
            img_array = np.array(raw_img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # 預測
            prediction = cnn_model.predict(img_array)
            result = np.argmax(prediction)
            confidence = np.max(prediction)
            
            st.success(f"🔍 AI 辨識結果：**{result}**")
            st.progress(float(confidence))
            st.write(f"信心程度：{confidence*100:.1f}%")

    with col2:
        st.subheader("🔬 AI 的視角：特徵圖")
        if uploaded_file:
            # 提取卷積層特徵
            layer_output = cnn_model.get_layer('conv_layer').output
            vis_model = tf.keras.models.Model(inputs=cnn_model.input, outputs=layer_output)
            features = vis_model.predict(img_array)
            
            # 畫出 4 個濾鏡的結果
            fig, axes = plt.subplots(2, 2, figsize=(6, 6))
            for i in range(4):
                ax = axes[i//2, i%2]
                ax.imshow(features[0, :, :, i], cmap='magma')
                ax.axis('off')
                ax.set_title(f"濾鏡 {i+1} 提取結果")
            st.pyplot(fig)
            st.markdown("> **老師筆記：** 你看！有的濾鏡在找水平線，有的在找圓弧。這就是 CNN 的『特徵提取』！")

# --- 模式 2：RNN 序列預測 ---
elif teaching_mode == "⏳ RNN 序列記憶力":
    st.title("⏳ RNN (循環神經網絡)")
    st.write("### 教學目標：理解 AI 如何「記住」先前的資訊。")

    # 解決你提到的記憶殘留問題：使用 Session State 顯式管理
    if 'rnn_history' not in st.session_state:
        st.session_state.rnn_history = []

    # 按鈕：重置記憶
    if st.sidebar.button("🧼 清除 AI 的筆記本"):
        st.session_state.rnn_history = []
        st.rerun()

    st.subheader("✍️ 實驗：輸入一個句子")
    user_input = st.text_input("輸入英文單字或句子 (例如: I am not happy):", key="rnn_input")

    if st.button("送入 AI 腦中"):
        if user_input:
            # 將輸入加入記憶
            st.session_state.rnn_history.append(user_input)

    # 視覺化呈現記憶鏈
    st.write("### 🧠 AI 目前的「記憶本」：")
    if st.session_state.rnn_history:
        # 用箭頭展示順序
        memory_chain = " ➔ ".join([f"[{word}]" for word in st.session_state.rnn_history])
        st.info(memory_chain)
        
        # 模擬情感分析結果
        score = np.random.random() # 這裡可以替換成真實模型
        st.write("---")
        st.subheader("📊 最終判斷結果")
        if "not" in " ".join(st.session_state.rnn_history).lower():
            st.error("😢 情感偵測：負面 (因為記憶中有 'not'，語意被反轉了)")
        else:
            st.success("😊 情感偵測：正面 (基於目前的記憶序列)")
            
        st.warning("⚠️ **觀察點：** 即使你現在輸入 'happy'，如果你的記憶本（上面那一串）前面有 'not'，AI 的判斷就會完全不同！這就是 RNN 的順序記憶。")
    else:
        st.write("目前記憶本是空的，請在上方輸入文字。")

# 4. 底部教學總結
st.markdown("---")
with st.expander("💡 國高中生必學總結"):
    st.write("""
    - **CNN (眼睛)**：不管東西在圖片的哪裡，只要「特徵」對了就認得。適合用在照片辨識。
    - **RNN (記憶)**：之前的資訊會影響現在的決定。適合用在翻譯、對話。
    - **隱藏狀態 (Hidden State)**：就是 AI 的小筆記本，記錄著前面看過的資訊。
    """)
