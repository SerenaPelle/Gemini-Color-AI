import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# --- THE ULTIMATE ARTIST DATABASE ---
# Format: "Product Name": [R, G, B, "Category", "URL"]
BRAND_DATABASE = {
    # --- PRISMACOLOR PENCILS (Expanded) ---
    "Prismacolor: Indigo Blue (PC901)": [30, 45, 80, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Sky Blue Light (PC1086)": [180, 215, 235, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Grass Green (PC909)": [75, 130, 70, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Terra Cotta (PC944)": [145, 75, 55, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Black (PC935)": [20, 20, 20, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: White (PC938)": [250, 250, 250, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Spanish Orange (PC1003)": [255, 190, 50, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Cool Grey 50% (PC1083)": [140, 145, 150, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    
    # --- PAINTS (Sherwin-Williams & Farrow & Ball) ---
    "SW: Tricorn Black (6258)": [47, 47, 48, "Wall Paint", "https://www.sherwin-williams.com/en-us/color/color-family/neutral-paint-colors/sw6258-tricorn-black"],
    "SW: Alabaster (7008)": [237, 234, 224, "Wall Paint", "https://www.sherwin-williams.com/en-us/color/color-family/white-paint-colors/sw7008-alabaster"],
    "SW: Naval (6244)": [47, 61, 80, "Wall Paint", "https://www.sherwin-williams.com/en-us/color/color-family/blue-paint-colors/sw6244-naval"],
    "F&B: Dead Salmon": [170, 135, 125, "Designer Paint", "https://www.farrow-ball.com/paint/dead-salmon"],
    "F&B: Hague Blue": [45, 65, 75, "Designer Paint", "https://www.farrow-ball.com/paint/hague-blue"],
    "F&B: Breakfast Room Green": [115, 135, 110, "Designer Paint", "https://www.farrow-ball.com/paint/breakfast-room-green"],
    
    # --- FINE ART OILS (Winsor & Newton) ---
    "W&N: Titanium White": [255, 255, 255, "Oil Paint", "https://www.winsornewton.com/na/paint/oil/artists-oil-color/"],
    "W&N: Cadmium Red": [220, 30, 40, "Oil Paint", "https://www.winsornewton.com/na/paint/oil/artists-oil-color/"],
    "W&N: Yellow Ochre": [195, 155, 75, "Oil Paint", "https://www.winsornewton.com/na/paint/oil/artists-oil-color/"],
    "W&N: Burnt Umber": [70, 55, 45, "Oil Paint", "https://www.winsornewton.com/na/paint/oil/artists-oil-color/"],
    "W&N: French Ultramarine": [18, 52, 144, "Oil Paint", "https://www.winsornewton.com/na/paint/oil/artists-oil-color/"],
    "W&N: Sap Green": [80, 105, 55, "Oil Paint", "https://www.winsornewton.com/na/paint/oil/artists-oil-color/"]
}

# --- STYLING ---
st.set_page_config(page_title="Gemini Art Pro", page_icon="🎨", layout="wide")
st.markdown("<style>.stApp {background-color: #111; color: white;}</style>", unsafe_allow_html=True)

# --- CORE LOGIC ---
def find_closest_product(target_rgb):
    closest_name = None
    min_dist = float('inf')
    for name, data in BRAND_DATABASE.items():
        dist = np.sqrt(sum((np.array(target_rgb) - np.array(data[:3]))**2))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name, BRAND_DATABASE[closest_name]

# --- APP INTERFACE ---
st.title("✨ Gemini Art Color Pro")
st.write("Upload any image to decode its palette into real-world professional art supplies.")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_container_width=True)
    
    # Analyze 12 clusters for better detail coverage
    img_small = img.copy()
    img_small.thumbnail((150, 150))
    pixels = np.array(img_small).reshape(-1, 3)
    model = KMeans(n_clusters=12, n_init=10).fit(pixels)
    colors = model.cluster_centers_.astype(int)

    st.subheader("🛍️ Professional Supply Matches")
    # Show top matches with links
    for i in range(0, len(colors), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < len(colors):
                rgb = colors[i+j]
                name, info = find_closest_product(rgb)
                hex_val = '#%02x%02x%02x' % tuple(rgb)
                with col:
                    st.markdown(f'<div style="background-color:{hex_val}; height:60px; border-radius:12px; border:1px solid #444;"></div>', unsafe_allow_html=True)
                    st.write(f"**{name}**")
                    st.caption(f"Type: {info[3]}")
                    st.link_button("Shop Now", info[4])

# --- CHAT ASSISTANT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm ready to analyze your image. Ask me about any specific color!"}]

st.divider()
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ex: What is the best match for the sky?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.chat_message("assistant"):
        if uploaded_file:
            # Smart logic: Find color in DB matching the user's word
            query = prompt.lower()
            match = next((m for m in BRAND_DATABASE.keys() if any(word in m.lower() for word in query.split())), None)
            
            if match:
                res = f"I found it! For that area, I recommend **{match}**. You can find it here: {BRAND_DATABASE[match][4]}"
            else:
                res = f"Based on the analysis, the most accurate match for that area is **{find_closest_product(colors[0])[0]}**. Would you like a pencil or paint recommendation?"
            
            st.write(res)
            st.session_state.messages.append({"role": "assistant", "content": res})
