import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# --- LOAD DATABASE ---
@st.cache_data
def load_data():
    return pd.read_csv('colors.csv')

try:
    df_colors = load_data()
except:
    st.error("Missing colors.csv file!")
    st.stop()

# --- CONFIG ---
st.set_page_config(page_title="Gemini Art Pro", layout="wide")

# --- MATH ENGINE ---
def find_best_match(target_rgb, dataframe):
    # This calculation finds the mathematical "distance" between colors
    r_diff = (dataframe['r'] - target_rgb[0])**2
    g_diff = (dataframe['g'] - target_rgb[1])**2
    b_diff = (dataframe['b'] - target_rgb[2])**2
    distances = np.sqrt(r_diff + g_diff + b_diff)
    
    # Sort by distance and pick the very first one
    temp_df = dataframe.copy()
    temp_df['dist'] = distances
    return temp_df.sort_values('dist').iloc[0]

# --- UI ---
st.title("🎨 Gemini Artist Assistant")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    img_small = image.copy()
    img_small.thumbnail((150, 150))
    pixels = np.array(img_small).reshape(-1, 3)
    model = KMeans(n_clusters=15, n_init=10).fit(pixels)
    colors = model.cluster_centers_.astype(int)

    st.subheader("🛍️ Detected Supply Matches")
    cols = st.columns(4)
    for i, rgb in enumerate(colors[:8]):
        match = find_best_match(rgb, df_colors)
        with cols[i % 4]:
            hex_v = '#%02x%02x%02x' % tuple(rgb)
            st.markdown(f'<div style="background-color:{hex_v}; height:40px; border-radius:5px;"></div>', unsafe_allow_html=True)
            st.markdown(f"**[{match['name']}]({match['url']})**")

# --- CHAT SYSTEM ---
st.divider()
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about the blue boat or red roof..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        if not uploaded_file:
            response = "Please upload an image first!"
        else:
            query = prompt.lower()
            families = {
                "red": ["red", "crimson", "terra", "vermilion", "rose", "pink", "magenta"],
                "blue": ["blue", "sky", "indigo", "cobalt", "navy", "cyan", "azure"],
                "green": ["green", "sage", "olive", "leaf", "emerald"],
                "grey": ["grey", "gray", "stone", "slate", "charcoal"],
                "brown": ["brown", "umber", "sienna", "wood", "earth"]
            }
            
            target = next((f for f, words in families.items() if any(w in query for w in words)), None)
            
            if target:
                # Filter CSV for that color family
                filtered_df = df_colors[df_colors['name'].str.lower().str.contains('|'.join(families[target]))].copy()
                
                if not filtered_df.empty:
                    # Find which image color is closest to this specific family
                    best_match = None
                    min_dist = 999
                    for rgb in colors:
                        m = find_best_match(rgb, filtered_df)
                        if m['dist'] < min_dist:
                            min_dist = m['dist']
                            best_match = m
                    response = f"I've identified the **{target}** tones! I recommend **{best_match['name']}**. [Shop here]({best_match['url']})"
                else:
                    response = f"I see you're looking for {target}, but I need more {target} shades in the CSV!"
            elif any(word in query for word in ["boat", "house", "water", "wall"]):
                top_3 = [find_best_match(c, df_colors)['name'] for c in colors[:3]]
                response = f"For the objects in this scene, the most prominent matches are **{top_3[0]}**, **{top_3[1]}**, and **{top_3[2]}**."
            else:
                response = f"I recommend **{find_best_match(colors[0], df_colors)['name']}** for that area."
        
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
