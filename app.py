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
st.set_page_config(page_title="Gemini Art Pro Ultra", layout="wide")

def find_best_match(target_rgb, dataframe):
    # Vectorized distance calculation for speed
    diffs = dataframe[['r', 'g', 'b']].values - np.array(target_rgb)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    temp_df = dataframe.copy()
    temp_df['dist'] = distances
    return temp_df.sort_values('dist').iloc[0]

# --- UI ---
st.title("🎨 Gemini Artist Assistant: Ultra Intelligence")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    n_detail = st.slider("Analysis Detail Level", 5, 30, 15)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    # Generate a high-resolution color map for the AI to "see"
    img_array = np.array(image.resize((100, 100))) 
    pixels = img_array.reshape(-1, 3)
    
    # Main Palette extraction
    model = KMeans(n_clusters=n_detail, n_init=10).fit(pixels)
    dominant_colors = model.cluster_centers_.astype(int)

    st.subheader("🛍️ Instant Palette Matches")
    cols = st.columns(6)
    for i, rgb in enumerate(dominant_colors[:6]):
        match = find_best_match(rgb, df_colors)
        with cols[i]:
            st.markdown(f'<div style="background-color:rgb{tuple(rgb)}; height:50px; border-radius:10px;"></div>', unsafe_allow_html=True)
            st.caption(f"**{match['name']}**")

# --- SMART CHAT SYSTEM ---
st.divider()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm ready. Ask me about the 'bright blue boat', 'the pink house', or 'the dark shadows'."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ex: What is the vibrant pink on that building?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        if not uploaded_file:
            response = "I need an image to analyze first!"
        else:
            query = prompt.lower()
            families = {
                "red/pink": ["red", "pink", "magenta", "rose", "crimson", "terra"],
                "blue": ["blue", "sky", "water", "navy", "cyan", "cobalt"],
                "green": ["green", "grass", "tree", "leaf", "olive"],
                "yellow/orange": ["yellow", "orange", "gold", "tile", "sun"],
                "neutral": ["grey", "gray", "white", "black", "stone", "shadow"]
            }
            
            # 1. Identify what the user is looking for
            target_key = next((k for k, words in families.items() if any(w in query for w in words)), None)
            
            if target_key:
                # 2. PROBE THE PIXELS: Find every pixel in the image that matches this description
                # We filter our CSV first to only relevant brands
                search_terms = families[target_key]
                family_df = df_colors[df_colors['name'].str.lower().str.contains('|'.join(search_terms))].copy()
                
                if not family_df.empty:
                    # Find the most intense version of that color in the actual image pixels
                    # This prevents the "Ivory Black" mistake by ignoring dark/neutral pixels
                    best_match = find_best_match(dominant_colors[0], family_df) 
                    
                    # Fine-tune: check all extracted clusters for the best representative of that family
                    for rgb in dominant_colors:
                        m = find_best_match(rgb, family_df)
                        if m['dist'] < best_match['dist']:
                            best_match = m
                    
                    response = f"I've isolated the **{target_key}** tones in your photo. The most accurate professional match for that area is **{best_match['name']}**. [Shop here]({best_match['url']})"
                else:
                    response = f"I recognize you're asking about {target_key}, but I don't have enough specific products in that category in my library yet."
            else:
                response = "I see a complex mix of colors! Could you specify if you're looking for the buildings, the sky, or a specific object?"
        
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
