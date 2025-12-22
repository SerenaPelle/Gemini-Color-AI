import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from skimage import color # Needs 'pip install scikit-image'

# --- LOAD DATABASE ---
@st.cache_data
def load_data():
    df = pd.read_csv('colors.csv')
    # Pre-calculate LAB values for the database for instant matching
    rgb_norm = df[['r', 'g', 'b']].values / 255.0
    df['lab'] = list(color.rgb2lab(rgb_norm.reshape(-1, 1, 3)).reshape(-1, 3))
    return df

try:
    df_colors = load_data()
except Exception as e:
    st.error(f"Error loading colors.csv: {e}")
    st.stop()

# --- MATH ENGINE (LAB SPACE) ---
def find_precise_match(target_rgb, dataframe):
    # Convert target to LAB
    target_norm = np.array(target_rgb).reshape(1, 1, 3) / 255.0
    target_lab = color.rgb2lab(target_norm).reshape(3)
    
    # Calculate Delta-E (Human-eye distance)
    db_labs = np.stack(dataframe['lab'].values)
    distances = np.sqrt(np.sum((db_labs - target_lab)**2, axis=1))
    
    best_idx = np.argmin(distances)
    return dataframe.iloc[best_idx], distances[best_idx]

# --- UI ---
st.set_page_config(page_title="Gemini Art Pro: Vision Lab", layout="wide")
st.title("🎨 Gemini Artist Assistant: LAB Vision")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    st.caption("Now using Delta-E Lab calculations for high-accuracy matching.")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    # Analyze image with high cluster count for detail
    img_small = image.resize((150, 150))
    pixels = np.array(img_small).reshape(-1, 3)
    model = KMeans(n_clusters=20, n_init=10).fit(pixels) # Increased to 20
    colors = model.cluster_centers_.astype(int)

    st.subheader("🛍️ Detected Professional Matches")
    cols = st.columns(6)
    for i, rgb in enumerate(colors[:12]):
        match, _ = find_precise_match(rgb, df_colors)
        with cols[i % 6]:
            hex_v = '#%02x%02x%02x' % tuple(rgb)
            st.markdown(f'<div style="background-color:{hex_v}; height:50px; border-radius:10px;"></div>', unsafe_allow_html=True)
            st.caption(f"**{match['name']}**")

# --- SMART CHAT SYSTEM ---
st.divider()
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ask about the orange house vs red house..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        if not uploaded_file:
            res = "Please upload an image first!"
        else:
            query = prompt.lower()
            # Strict definitions to stop the 'Rosemary' (Green) or 'Dovetail' (Grey) errors
            families = {
                "red": ["red", "house", "roof", "building", "crimson", "terra"],
                "orange": ["orange", "wall", "terracotta", "clay", "amber"],
                "blue": ["blue", "sky", "water", "navy", "boat"],
                "green": ["green", "tree", "grass", "shutter", "olive"]
            }
            
            target = next((f for f, words in families.items() if any(w in query for w in words)), None)
            obj = next((w for w in ["house", "boat", "car", "sky", "wall", "roof"] if w in query), "area")
            
            if target:
                # 1. FORCE FILTER: Only allow shades from the CSV that match the word
                filtered_df = df_colors[df_colors['name'].str.lower().str.contains('|'.join(families[target]))].copy()
                
                if not filtered_df.empty:
                    best_match = None
                    min_dist = 999
                    
                    # 2. VIBRANCY SCAN: Look for the most 'intense' version of that color
                    for rgb in colors:
                        # Check saturation: (Max - Min). Greys/Shadows have low saturation.
                        sat = np.max(rgb) - np.min(rgb)
                        
                        # Only proceed if the pixel is actually colorful
                        if sat > 25: 
                            m, d = find_precise_match(rgb, filtered_df)
                            # We want the color that is mathematically closest to our target family
                            if d < min_dist:
                                min_dist = d
                                best_match = m
                    
                    if best_match is not None:
                        res = f"I've isolated the **{target}** from the background. For that specific {obj}, the best match is **{best_match['name']}**. [Link to Shade]({best_match['url']})"
                    else:
                        # Fallback if the image is too dark/shadowy
                        res = f"The {obj} is in deep shadow, but based on the hue, I recommend **{filtered_df.iloc[0]['name']}**."
                else:
                    res = f"I see the {target}, but I need more {target} shades in your CSV to be 100% sure!"
            else:
                top_match, _ = find_precise_match(colors[0], df_colors)
                res = f"The most prominent match for that {obj} is **{top_match['name']}**."
        
        st.write(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
