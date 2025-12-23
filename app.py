import streamlit as st
from PIL import Image, ImageDraw
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

def find_precise_match(target_rgb, dataframe):
    target_norm = np.array(target_rgb).reshape(1, 1, 3) / 255.0
    target_lab = color.rgb2lab(target_norm).reshape(3)
    db_labs = np.stack(dataframe['lab'].values)
    distances = np.sqrt(np.sum((db_labs - target_lab)**2, axis=1))
    return dataframe.iloc[np.argmin(distances)]

# --- UI SETUP ---
st.set_page_config(page_title="Pro-Palette & Picker", layout="wide")
st.title("🎯 Pro-Palette & Picker")

if uploaded_file := st.sidebar.file_uploader("Upload Scene", type=["jpg", "png"]):
    img = Image.open(uploaded_file).convert('RGB')
    width, height = img.size
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Interactive Picker")
        # Coordinate Sliders
        x = st.slider("Left ↔ Right", 0, width, width // 2)
        y = st.slider("Top ↔ Bottom", 0, height, height // 2)
        
        # Get the exact color at that pixel
        pixel_rgb = img.getpixel((x, y))
        match = find_precise_match(pixel_rgb, df_colors)
        
        st.divider()
        st.subheader("🛍️ Matching Product")
        st.markdown(f'<div style="background-color:rgb{pixel_rgb}; height:60px; border-radius:10px; border:2px solid white; margin-bottom:10px;"></div>', unsafe_allow_html=True)
        st.write(f"**Name:** {match['name']}")
        st.write(f"**Type:** {match['category']}")
        st.markdown(f"[Shop this Shade ↗️]({match['url']})")
        
    with col1:
        # 1. DRAW THE POINTER
        # We create a copy of the image and draw a red targeting circle on it
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        radius = 20
        # Draw the 'Crosshair' circle
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="red", width=5)
        # Draw a small center dot
        draw.ellipse((x-2, y-2, x+2, y+2), fill="red")
        
        st.image(draw_img, use_container_width=True)
        st.caption(f"Currently analyzing pixel at: {x}, {y}")

# --- CHAT ASSISTANT ---
st.divider()
if prompt := st.chat_input("Ask about this specific color..."):
    with st.chat_message("assistant"):
        st.write(f"I see you've targeted a specific area. That exact shade is closest to **{match['name']}**. Because this is a **{match['category']}**, it will have the right finish for the building you are painting.")
