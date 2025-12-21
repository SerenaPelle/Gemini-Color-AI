import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# --- BRAND DATABASE (same as before) ---
BRAND_DATABASE = {
    "Sherwin-Williams: Naval": [47, 61, 80, "Paint", "https://www.sherwin-williams.com"],
    "Prismacolor: Indigo Blue (PC901)": [30, 45, 80, "Pencil", "https://www.prismacolor.com"],
    "Winsor & Newton: French Ultramarine": [18, 52, 144, "Oil Paint", "https://www.winsornewton.com"],
    "Faber-Castell: Emerald Green": [0, 150, 100, "Pencil", "https://www.fabercastell.com"],
}

st.set_page_config(page_title="Gemini Art Assistant", page_icon="🎨", layout="wide")

# --- INITIALIZE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your color assistant. Upload a photo, and I can help you find the perfect art supplies!"}
    ]

# --- FUNCTIONS ---
def find_closest_product(target_rgb):
    closest_name = None
    min_dist = float('inf')
    for name, data in BRAND_DATABASE.items():
        dist = np.sqrt(sum((np.array(target_rgb) - np.array(data[:3]))**2))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name, BRAND_DATABASE[closest_name]

# --- MAIN APP UI ---
st.title("✨ Gemini Color AI + Assistant")

# Sidebar for Image Upload
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Current Reference", use_container_width=True)
    
    # Process 3 dominant colors
    img_small = image.copy().convert('RGB')
    img_small.thumbnail((100, 100))
    pixels = np.array(img_small).reshape(-1, 3)
    model = KMeans(n_clusters=3, n_init=10).fit(pixels)
    colors = model.cluster_centers_.astype(int)

    st.write("### 🛍️ Suggested Supplies")
    cols = st.columns(3)
    found_matches = []
    for i, rgb in enumerate(colors):
        hex_val = '#%02x%02x%02x' % tuple(rgb)
        product_name, info = find_closest_product(rgb)
        found_matches.append(product_name)
        with cols[i]:
            st.markdown(f'<div style="background-color:{hex_val}; height:40px; border-radius:10px;"></div>', unsafe_allow_html=True)
            st.caption(f"**{product_name}**")

# --- CHAT INTERFACE ---
st.divider()
st.subheader("💬 Ask Your Assistant")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ex: 'Which blue is best for a sky?'"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate "AI" response
    with st.chat_message("assistant"):
        if uploaded_file:
            response = f"Based on your photo, I recommend focusing on **{found_matches[0]}**. It's the most dominant shade I found!"
        else:
            response = "Please upload an image so I can see the colors you're working with!"
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
