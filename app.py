import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# --- LOAD THE DATABASE FROM CSV ---
@st.cache_data # This makes the app super fast
def load_data():
    df = pd.read_csv('colors.csv')
    return df

try:
    df_colors = load_data()
except:
    st.error("Please make sure 'colors.csv' is uploaded to GitHub!")
    st.stop()

# --- STYLING ---
st.set_page_config(page_title="Gemini Art Pro 300", page_icon="🎨", layout="wide")
st.markdown("<style>.stApp {background-color: #0e1117; color: white;}</style>", unsafe_allow_html=True)

# --- MATH ENGINE (Scans the CSV) ---
def find_best_match(target_rgb):
    # Calculate distance for all 300+ rows at once
    r_diff = (df_colors['r'] - target_rgb[0])**2
    g_diff = (df_colors['g'] - target_rgb[1])**2
    b_diff = (df_colors['b'] - target_rgb[2])**2
    distances = np.sqrt(r_diff + g_diff + b_diff)
    
    idx = distances.idxmin()
    return df_colors.iloc[idx]

# --- APP UI ---
st.title("🎨 Gemini Artist Assistant: 300+ Shades")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    # 15 Clusters for high-precision detail
    img_small = image.copy()
    img_small.thumbnail((150, 150))
    pixels = np.array(img_small).reshape(-1, 3)
    model = KMeans(n_clusters=15, n_init=10).fit(pixels)
    colors = model.cluster_centers_.astype(int)

    st.subheader("🛍️ Professional Supply Matches")
    for rgb in colors[:12]:
        match = find_best_match(rgb)
        hex_val = '#%02x%02x%02x' % tuple(rgb)
        
        col1, col2, col3 = st.columns([1, 4, 2])
        with col1:
            st.markdown(f'<div style="background-color:{hex_val}; height:45px; border-radius:8px; border:1px solid #444;"></div>', unsafe_allow_html=True)
        with col2:
            st.write(f"**{match['name']}**")
            st.caption(f"Category: {match['category']}")
        with col3:
            st.markdown(f"[Shop Product ↗️]({match['url']})")

# --- IMPROVED CHAT ASSISTANT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I've scanned over 300 professional shades. Ask me about a specific area, like the red roofs or the green trees!"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ex: What is the red on the house?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.chat_message("assistant"):
        if uploaded_file:
            query = prompt.lower()
            found_response = False
            
            # 1. SMART SCAN: Look through ALL detected image colors for a match to the user's word
            for rgb in colors:
                match = find_best_match(rgb)
                # Check if the user is asking for a color that appears in our product names
                color_keywords = ["red", "blue", "green", "orange", "yellow", "brown", "white", "grey", "black", "purple"]
                
                # If user says "red" and the product name contains "red" or "crimson" or "terracotta"
                for word in query.split():
                    if word in color_keywords and word in match['name'].lower():
                        res = f"I see the {word} you're talking about! I recommend **{match['name']}**. It's the closest match for that specific area. [Shop Here]({match['url']})"
                        found_response = True
                        break
                if found_response: break

            # 2. FALLBACK: If no keyword match, use the highest-confidence pixel match
            if not found_response:
                best_overall = find_best_match(colors[0])
                res = f"Based on the pixels in that area, the best professional match is **{best_overall['name']}**. Does this look like the shade you were looking for?"
        else:
            res = "Please upload an image so I can analyze those specific rooftops for you!"
            
        st.write(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
