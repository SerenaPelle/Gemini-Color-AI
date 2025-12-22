import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# --- LOAD DATABASE ---
@st.cache_data
def load_data():
    # Make sure colors.csv is in your GitHub folder!
    return pd.read_csv('colors.csv')

try:
    df_colors = load_data()
except:
    st.error("Missing colors.csv file!")
    st.stop()

# --- CONFIG ---
st.set_page_config(page_title="Gemini Art Pro", layout="wide")

def find_best_match(target_rgb, dataframe=df_colors):
    r_diff = (dataframe['r'] - target_rgb[0])**2
    g_diff = (dataframe['g'] - target_rgb[1])**2
    b_diff = (dataframe['b'] - target_rgb[2])**2
    distances = np.sqrt(r_diff + g_diff + b_diff)
    return dataframe.iloc[distances.idxmin()]

# --- UI ---
st.title("🎨 Gemini Artist Assistant")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    # Analyze image for 15 specific clusters
    img_small = image.copy()
    img_small.thumbnail((150, 150))
    pixels = np.array(img_small).reshape(-1, 3)
    model = KMeans(n_clusters=15, n_init=10).fit(pixels)
    colors = model.cluster_centers_.astype(int)

    st.subheader("🛍️ Detected Supply Matches")
    cols = st.columns(4)
    for i, rgb in enumerate(colors[:8]):
        match = find_best_match(rgb)
        with cols[i % 4]:
            hex_v = '#%02x%02x%02x' % tuple(rgb)
            st.markdown(f'<div style="background-color:{hex_v}; height:40px; border-radius:5px;"></div>', unsafe_allow_html=True)
            st.markdown(f"**[{match['name']}]({match['url']})**")

# --- CHAT (Fixed Indentation) ---
st.divider()
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about the red on the roof..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        if not uploaded_file:
            response = "Please upload an image first!"
        else:
            query = prompt.lower()
            # Define color families to help the AI "filter" its brain
            families = {
                "red": ["red", "crimson", "terra", "vermilion", "rose"],
                "blue": ["blue", "sky", "indigo", "cobalt", "navy"],
                "green": ["green", "sage", "olive", "leaf"]
            }
            
            # Check if user mentioned a color
            target = next((f for f, words in families.items() if any(w in query for w in words)), None)
            
            if target:
                # Filter CSV for that color family
                filtered_df = df_colors[df_colors['name'].str.lower().str.contains('|'.join(families[target]))]
                if not filtered_df.empty:
                    # Find which of the 15 clusters is closest to the family
                    best_match = None
                    min_dist = 999
                    for rgb in colors:
                        m = find_best_match(rgb, filtered_df)
                        dist = np.sqrt(sum((np.array(rgb) - np.array(m[['r','g','b']]))**2))
                        if dist < min_dist:
                            min_dist = dist
                            best_match = m
                    response = f"I found the {target}! I recommend **{best_match['name']}**. [Shop here]({best_match['url']})"
                else:
                    response = f"I see you want {target}, but I need more {target} shades in the CSV!"
            else:
                response = f"I recommend **{find_best_match(colors[0])['name']}** for the main area."
        
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
