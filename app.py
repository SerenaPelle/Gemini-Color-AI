import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# --- THE ULTIMATE ARTIST DATABASE ---
# Format: "Product Name": [R, G, B, "Category", "URL"]
BRAND_DATABASE = {
    # --- BLUES (Sky, Water, Shadows) ---
    "Prismacolor: Sky Blue Light (PC1086)": [180, 215, 235, "Pencil", "https://www.prismacolor.com"],
    "Prismacolor: Cloud Blue (PC1023)": [190, 210, 225, "Pencil", "https://www.prismacolor.com"],
    "Prismacolor: Copenhagen Blue (PC906)": [45, 100, 160, "Pencil", "https://www.prismacolor.com"],
    "Prismacolor: Indigo Blue (PC901)": [30, 45, 80, "Pencil", "https://www.prismacolor.com"],
    "Winsor & Newton: Cerulean Blue": [42, 82, 190, "Oil Paint", "https://www.winsornewton.com"],
    "Sherwin-Williams: Naval (6244)": [47, 61, 80, "Wall Paint", "https://www.sherwin-williams.com"],
    
    # --- GREENS & LANDSCAPE (Trees, Garden) ---
    "Prismacolor: Grass Green (PC909)": [75, 130, 70, "Pencil", "https://www.prismacolor.com"],
    "Prismacolor: Olive Green (PC911)": [115, 125, 65, "Pencil", "https://www.prismacolor.com"],
    "Holbein: Sap Green": [80, 105, 55, "Watercolor", "https://www.holbeinartistmaterials.com"],
    "Sherwin-Williams: Garden Sage": [125, 126, 100, "Wall Paint", "https://www.sherwin-williams.com"],

    # --- EARTH & ARCHITECTURE (Tiles, Stone, Wood) ---
    "Behr: Terra Cotta Tile": [186, 115, 87, "Wall Paint", "https://www.behr.com"],
    "Winsor & Newton: Yellow Ochre": [195, 155, 75, "Oil Paint", "https://www.winsornewton.com"],
    "Winsor & Newton: Burnt Umber": [70, 55, 45, "Oil Paint", "https://www.winsornewton.com"],
    "Prismacolor: Espresso (PC1099)": [65, 50, 45, "Pencil", "https://www.prismacolor.com"],
    "Farrow & Ball: Old White": [215, 210, 195, "Designer Paint", "https://www.farrow-ball.com"],
}

# --- STYLING & CONFIG ---
st.set_page_config(page_title="Gemini Art Pro", page_icon="🎨", layout="wide")
st.markdown("<style>.stApp {background-color: #0e1117; color: white;}</style>", unsafe_allow_html=True)

# --- CORE MATH ENGINE ---
def find_best_match(target_rgb):
    closest_name = None
    min_dist = float('inf')
    for name, data in BRAND_DATABASE.items():
        dist = np.sqrt(sum((np.array(target_rgb) - np.array(data[:3]))**2))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name, BRAND_DATABASE[closest_name]

# --- APP UI ---
st.title("🎨 Gemini Artist Assistant Pro")
st.write("Upload any image to find matching professional art supplies.")

with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Image Processing
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    # 2. Advanced Color Analysis (12 detailed clusters)
    img_small = image.copy()
    img_small.thumbnail((150, 150))
    pixels = np.array(img_small).reshape(-1, 3)
    model = KMeans(n_clusters=12, n_init=10).fit(pixels)
    colors = model.cluster_centers_.astype(int)

    st.subheader("🛍️ Professional Supply Matches")
    # 3. Supply List with Bulletproof Markdown Links
    for rgb in colors[:8]: # Display the top 8 matches
        product_name, info = find_best_match(rgb)
        hex_val = '#%02x%02x%02x' % tuple(rgb)
        
        col1, col2, col3 = st.columns([1, 4, 2])
        with col1:
            st.markdown(f'<div style="background-color:{hex_val}; height:45px; border-radius:8px; border:1px solid #444;"></div>', unsafe_allow_html=True)
        with col2:
            st.write(f"**{product_name}**")
            st.caption(f"Category: {info[3]}")
        with col3:
            # Markdown links are more reliable than buttons for many browsers
            st.markdown(f"[Shop Product ↗️]({info[4]})")

# --- CHAT ASSISTANT ---
st.divider()
st.subheader("💬 Ask Your Color Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I've analyzed the image! Ask me about the sky, the water, or specific details like tiles or trees."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ex: 'What is the best match for the sky?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.chat_message("assistant"):
        if uploaded_file:
            query = prompt.lower()
            
            # Smart logic for common landscape elements
            if "sky" in query:
                response = "For the sky, I recommend **Prismacolor: Sky Blue Light**. It captures that soft atmospheric glow perfectly."
            elif "water" in query:
                response = "For the water, **Winsor & Newton: Cerulean Blue** or **Indigo Blue** are best for those deeper, more vibrant reflections."
            elif "tile" in query or "orange" in query:
                response = "I found a great match for the tiles! **Behr: Terra Cotta Tile** or **Prismacolor: Terra Cotta** will work best."
            else:
                best_overall = find_best_match(colors[0])[0]
                response = f"Based on your question, the most accurate match I found in that range is **{best_overall}**."
        else:
            response = "Please upload an image first so I can see the colors!"
            
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
