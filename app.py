import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# --- 頁面配置 ---
st.set_page_config(page_title="CNN 理論實驗室 (李宏毅老師教材版)", layout="wide")

st.title("🖼️ CNN 深度解析：從影像到下圍棋")
st.markdown("""
本工具根據 **李宏毅老師《深度學習详解》第四章** 設計。
CNN 的核心思維：**不需要看整張圖，只要看局部特徵就能辨識物體。**
""")

tab1, tab2, tab3 = st.tabs(["🧩 卷積理論 (可調參數)", "🎯 影像特徵視覺化", "⚪ AlphaGo 下圍棋原理"])

# =================================================================
# Tab 1: 卷積理論 (感受野與參數共享)
# =================================================================
with tab1:
    st.header("💡 卷積層的核心參數")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### ⚙️ 調整參數 (Hyperparameters)")
        kernel_size = st.slider("卷積核大小 (Kernel Size)", 1, 5, 3, help="對應感受野的大小")
        stride = st.slider("步幅 (Stride)", 1, 3, 1, help="滑動的距離")
        padding = st.selectbox("填充 (Padding)", ["Valid (無填充)", "Same (維持原圖大小)"])
        
    with col2:
        st.write("### 🔍 矩陣運算視覺化")
        # 創建一個模擬的 8x8 圖像矩陣
        img_sim = np.zeros((8, 8))
        img_sim[2:6, 2:6] = 255 # 中間一個方塊
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img_sim, cmap='gray')
        # 畫出感受野 (Receptive Field) 的框
        rect = plt.Rectangle((1, 1), kernel_size, kernel_size, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"感受野 (Red Box) 正在掃描圖片")
        st.pyplot(fig)
        
    st.info(f"**理論重點：** 我們不需要每個神經元都看整張圖。這個 {kernel_size}x{kernel_size} 的紅框就是一個神經元的『感受野』，它只負責偵測這部分的特徵。")
    

# =================================================================
# Tab 2: 影像特徵與關注區域 (Grad-CAM)
# =================================================================
with tab2:
    st.header("🖼️ 特徵提取：AI 到底在看什麼？")
    up_file = st.file_uploader("上傳照片 (貓/狗/車...)", type=['jpg', 'png'])
    
    if up_file:
        img = Image.open(up_file).convert('RGB').resize((224, 224))
        img_arr = np.array(img)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img, caption="原始影像")
        with c2:
            st.write("🔍 **邊緣偵測特徵**")
            # 模擬卷積核處理
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            st.image(edges, caption="CNN 前幾層看到的線條")
        with c3:
            st.write("🔥 **熱力關注圖 (Grad-CAM)**")
            # 模擬 Heatmap
            heatmap = np.random.rand(224, 224)
            fig, ax = plt.subplots()
            ax.imshow(img); ax.imshow(heatmap, cmap='jet', alpha=0.5); ax.axis('off')
            st.pyplot(fig)
            
    st.markdown("> **書中觀點：** CNN 的後層會把前層抓到的『線條』組合成『器官』或『物體輪廓』。")
    

# =================================================================
# Tab 3: CNN 應用於下圍棋 (AlphaGo)
# =================================================================
with tab3:
    st.header("⚪ AlphaGo：為什麼棋盤可以當成圖片？")
    st.write("根據李宏毅老師教材，圍棋的棋盤 (19x19) 可以被視為一張 **19x19 像素、有 48 個通道** 的圖片。")
    
    col_go1, col_go2 = st.columns(2)
    
    with col_go1:
        # 繪製簡單棋盤
        go_board = np.zeros((19, 19, 3)) + 0.8 # 木色背景
        # 隨機放幾顆棋子
        go_board[3, 3] = [0, 0, 0]; go_board[15, 15] = [1, 1, 1] 
        fig, ax = plt.subplots()
        ax.imshow(go_board); ax.set_xticks(range(19)); ax.set_yticks(range(19)); ax.grid(True)
        st.pyplot(fig)
        st.caption("19x19 的圍棋盤面 = 特殊的影像輸入")

    with col_go2:
        st.write("### 為什麼圍棋適合用 CNN？")
        st.success("1. **局部性 (Locality)**：判斷一個棋子是否被『吃掉』，只需要看它四周的棋子，不需要看整個棋盤。")
        st.success("2. **平移不變性 (Translation Invariance)**：左上角的『吃子手筋』跟右下角的道理是一樣的，可以共享參數。")
        
    st.info("💡 **小知識：** AlphaGo 的 CNN 並沒有使用『池化層 (Pooling)』，因為在圍棋中，棋子的精確位置差一格就差很多，不能隨便壓縮尺寸！")
    

# --- 全域清除 ---
if st.sidebar.button("🧼 清除所有緩存"):
    st.session_state.clear()
    st.rerun()
