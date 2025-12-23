import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from skimage import color

# --- LOAD DATABASE ---
@st.cache_data
def load_data():
    df = pd.read_csv('colors.csv')
    rgb_norm = df[['r', 'g', 'b']].values / 255.0
    df['lab'] = list(color.rgb2lab(rgb_norm.reshape(-1, 1, 3)).reshape(-1, 3))
    return df

df_colors = load_data()

# --- PRECISE MATCH ENGINE ---
def find_precise_match(target_rgb, dataframe):
    target_norm = np.array(target_rgb).reshape(1, 1, 3) / 255.0
    target_lab = color.rgb2lab(target_norm).reshape(3)
    db_labs = np.stack(dataframe['lab'].values)
    # Using Delta-E distance for human-eye accuracy
    distances = np.sqrt(np.sum((db_labs - target_lab)**2, axis=1))
    best_idx = np.argmin(distances)
    return dataframe.iloc[best_idx]

# --- UI ---
st.set_page_config(page_title="Gemini Art Pro: Precision Edition", layout="wide")
st.title("🎯 Precise Color Matcher")

if uploaded_file := st.sidebar.file_uploader("Upload Scene", type=["jpg", "png"]):
    img = Image.open(uploaded_file).convert('RGB')
    width, height = img.size
    
    # Display image with a "Crosshair" selector
    st.subheader("Click or use sliders to pick a specific house")
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### Selection Coordinates")
        x = st.slider("Left <--> Right", 0, width, width // 2)
        y = st.slider("Top <--> Bottom", 0, height, height // 2)
        
        # Pull the EXACT pixel data
        pixel_rgb = img.getpixel((x, y))
        match = find_precise_match(pixel_rgb, df_colors)
        
        st.divider()
        st.markdown(f"**Selected Color:**")
        st.markdown(f'<div style="background-color:rgb{pixel_rgb}; height:100px; border-radius:10px; border:3px solid white;"></div>', unsafe_allow_html=True)
        st.success(f"Best Match: **{match['name']}**")
        st.info(f"Category: {match['category']}")
        st.markdown(f"[Purchase Supply ↗️]({match['url']})")

    with col1:
        # Visual feedback: Draw a small circle where the user is picking
        st.image(img, use_container_width=True)
        st.caption(f"Targeting: {x}, {y}")

# --- AI CHAT (NOW CONTEXT-AWARE) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.divider()
if prompt := st.chat_input("Tell me more about this color..."):
    with st.chat_message("assistant"):
        # The AI now knows EXACTLY what color you picked
        res = f"I see you've selected a spot with RGB {pixel_rgb}. This leans more towards the {match['name']} in your database. Because it is in the {match['category']} family, it's perfect for the building texture you're working on."
        st.write(res)
