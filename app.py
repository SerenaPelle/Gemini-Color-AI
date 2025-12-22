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

# --- CHAT ASSISTANT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm searching 300+ professional shades. What can I find for you?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ask about a color..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.chat_message("assistant"):
        if uploaded_file:
            # Smart logic looks through the CSV for keywords
            query = prompt.lower()
            keyword_match = df_colors[df_colors['name'].str.contains(query, case=False, na=False)]
            
            if not keyword_match.empty:
                top_hit = keyword_match.iloc[0]
                res = f"From our 300+ shades, I found **{top_hit['name']}**. It's a perfect match for your description! [Link]({top_hit['url']})"
            else:
                res = f"I've analyzed that. Based on the pixels, the closest professional match is **{find_best_match(colors[0])['name']}**."
        else:
            res = "Please upload an image so I can scan the database!"
        st.write(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
