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
    if dataframe.empty:
        return None
    target_norm = np.array(target_rgb).reshape(1, 1, 3) / 255.0
    target_lab = color.rgb2lab(target_norm).reshape(3)
    db_labs = np.stack(dataframe['lab'].values)
    distances = np.sqrt(np.sum((db_labs - target_lab)**2, axis=1))
    return dataframe.iloc[np.argmin(distances)]

# --- UI SETUP ---
st.set_page_config(page_title="Gemini Art Pro: Multi-Medium", layout="wide")
st.title("🎯 Pro-Palette: Pencil, Oil & Acrylic")

# --- SIDEBAR FILTERS ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload Scene", type=["jpg", "png"])
    st.divider()
    st.subheader("Filter by Medium")
    # This allows users to search only for specific art supplies
    medium_choice = st.multiselect(
        "Select allowed supplies:",
        options=["Pencil", "Oil Paint", "Acrylic Paint", "Wall Paint"],
        default=["Pencil", "Oil Paint", "Acrylic Paint", "Wall Paint"]
    )

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    width, height = img.size
    
    # Filter the database based on the sidebar selection
    active_df = df_colors[df_colors['category'].isin(medium_choice)]
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Interactive Picker")
        x = st.slider("Left ↔ Right", 0, width, width // 2)
        y = st.slider("Top ↔ Bottom", 0, height, height // 2)
        
        pixel_rgb = img.getpixel((x, y))
        
        if not active_df.empty:
            match = find_precise_match(pixel_rgb, active_df)
            
            st.divider()
            st.subheader("🛍️ Precise Match")
            st.markdown(f'<div style="background-color:rgb{pixel_rgb}; height:60px; border-radius:10px; border:2px solid white;"></div>', unsafe_allow_html=True)
            st.write(f"**Name:** {match['name']}")
            st.write(f"**Medium:** {match['category']}")
            st.markdown(f"[Shop this Shade ↗️]({match['url']})")
        else:
            st.warning("Please select at least one medium in the sidebar!")
        
    with col1:
        # Crosshair logic
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        radius = 25
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="red", width=8)
        st.image(draw_img, use_container_width=True)

# --- SMART CHAT (AUTO-FILTERING) ---
st.divider()
if prompt := st.chat_input("Ask about the red house colors..."):
    with st.chat_message("assistant"):
        query = prompt.lower()
        # The AI now prioritizes the user's filtered medium
        if active_df.empty:
            st.write("I need you to select a medium (Pencils, Oils, etc.) in the sidebar first!")
        else:
            # We look for color keywords (red, orange, etc.)
            colors_map = {"red": ["red", "crimson"], "orange": ["orange", "terra"]}
            target_hue = next((k for k, v in colors_map.items() if any(word in query for word in v)), None)
            
            if target_hue:
                # Filter specifically for the color family + the chosen medium
                hue_df = active_df[active_df['name'].str.lower().str.contains(target_hue)]
                if not hue_df.empty:
                    chat_match = find_precise_match(pixel_rgb, hue_df)
                    st.write(f"In your selected **{', '.join(medium_choice)}** range, the best match for that {target_hue} area is **{chat_match['name']}**.")
                else:
                    st.write(f"I found the color, but I don't have any **{target_hue}** supplies in the **{', '.join(medium_choice)}** category yet.")
            else:
                st.write(f"The exact match for that spot in your selected media is **{match['name']}**.")
