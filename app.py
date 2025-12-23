import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import json
from skimage import color

# --- LOAD DATA & KNOWLEDGE ---
@st.cache_data
def load_resources():
    # Load Main Colors
    df = pd.read_csv('colors.csv')
    rgb_norm = df[['r', 'g', 'b']].values / 255.0
    df['lab'] = list(color.rgb2lab(rgb_norm.reshape(-1, 1, 3)).reshape(-1, 3))
    
    # Load AI Knowledge
    with open('knowledge_base.json', 'r') as f:
        kb = json.load(f)
    return df, kb

df_colors, art_kb = load_resources()

# --- THE "SMART" MATCH ENGINE ---
def find_precise_match(target_rgb, dataframe):
    target_norm = np.array(target_rgb).reshape(1, 1, 3) / 255.0
    target_lab = color.rgb2lab(target_norm).reshape(3)
    db_labs = np.stack(dataframe['lab'].values)
    # Using Delta-E for professional accuracy
    distances = np.sqrt(np.sum((db_labs - target_lab)**2, axis=1))
    return dataframe.iloc[np.argmin(distances)]

# --- UI SETUP ---
st.set_page_config(page_title="Gemini Art Pro: Expert Mode", layout="wide")
st.title("🎨 Gemini Expert Artist Assistant")

# Sidebar for precise control
with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    st.divider()
    selected_mediums = st.multiselect(
        "Preferred Media", 
        ["Pencil", "Oil Paint", "Acrylic Paint", "Wall Paint"],
        default=["Pencil", "Oil Paint", "Acrylic Paint"]
    )

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    width, height = img.size
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🎯 Selection & Match")
        x = st.slider("X (Horizontal)", 0, width, width // 2)
        y = st.slider("Y (Vertical)", 0, height, height // 2)
        
        pixel_rgb = img.getpixel((x, y))
        active_df = df_colors[df_colors['category'].isin(selected_mediums)]
        match = find_precise_match(pixel_rgb, active_df)
        
        # Display selection color
        st.markdown(f'<div style="background-color:rgb{pixel_rgb}; height:60px; border-radius:10px; border:2px solid white;"></div>', unsafe_allow_html=True)
        st.success(f"**{match['name']}**")
        st.info(f"Expert Tip: {art_kb['medium_characteristics'].get(match['category'], 'Great for this scene!')}")
        st.markdown(f"[Shop Product ↗️]({match['url']})")

    with col1:
        # Crosshair Pointer
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        draw.ellipse((x-20, y-20, x+20, y+20), outline="red", width=5)
        st.image(draw_img, use_container_width=True)

# --- THE SMARTER CHAT (REASONING ENGINE) ---
st.divider()
if prompt := st.chat_input("Ex: Why is the shadow on the red house looking purple?"):
    st.session_state.messages = st.session_state.get("messages", [])
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        query = prompt.lower()
        
        # AI uses art_kb to answer smartly
        if "shadow" in query:
            reasoning = art_kb['color_theory']['shadow_logic']
            res = f"Great observation! {reasoning} For that specific shadow, I recommend using **{match['name']}** as a base."
        elif "orange" in query or "red" in query:
            res = f"I've analyzed the building. To capture that Venetian heat, I'm matching your selected point to **{match['name']}**. {art_kb['medium_characteristics'][match['category']]}"
        else:
            res = f"Based on your selection, I recommend **{match['name']}**. This fits within the classic **{', '.join(art_kb['color_theory']['venetian_palette'])}** palette found in this type of architecture."
        
        st.write(res)
