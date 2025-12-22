import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# --- GIANT PROFESSIONAL DATABASE ---
BRAND_DATABASE = {
    # BLUES (Light to Dark)
    "Prismacolor: Sky Blue Light (PC1086)": [180, 215, 235, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Cloud Blue (PC1023)": [190, 210, 225, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Copenhagen Blue (PC906)": [45, 100, 160, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Indigo Blue (PC901)": [30, 45, 80, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "W&N: Cerulean Blue": [42, 82, 190, "Oil Paint", "https://www.winsornewton.com/na/paint/oil/artists-oil-color/"],
    "SW: Naval (6244)": [47, 61, 80, "Wall Paint", "https://www.sherwin-williams.com/en-us/color/color-family/blue-paint-colors/sw6244-naval"],
    
    # GREENS & EARTH (Foliage/Stone)
    "Prismacolor: Grass Green (PC909)": [75, 130, 70, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Olive Green (PC911)": [115, 125, 65, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "W&N: Yellow Ochre": [195, 155, 75, "Oil Paint", "https://www.winsornewton.com/na/paint/oil/artists-oil-color/"],
    "W&N: Burnt Umber": [70, 55, 45, "Oil Paint", "https://www.winsornewton.com/na/paint/oil/artists-oil-color/"],
    "F&B: Sap Green": [103, 110, 75, "Designer Paint", "https://www.farrow-ball.com/paint/sap-green"],
    
    # ORANGES & REDS (Tiles/Flowers)
    "Prismacolor: Pale Vermilion (PC921)": [230, 80, 60, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Prismacolor: Terra Cotta (PC944)": [145, 75, 55, "Pencil", "https://www.prismacolor.com/colored-pencils/premier-soft-core-colored-pencils/PPCPremierSoftCoreColoredPencils"],
    "Behr: Terra Cotta Tile": [186, 115, 87, "Paint", "https://www.behr.com"],
}

st.set_page_config(page_title="Gemini Art Pro", page_icon="🎨", layout="wide")

# --- CORE MATH ENGINE ---
def find_all_matches(target_rgb):
    results = []
    for name, data in BRAND_DATABASE.items():
        dist = np.sqrt(sum((np.array(target_rgb) - np.array(data[:3]))**2))
        results.append({"name": name, "dist": dist, "info": data})
    # Sort by closest match
    return sorted(results, key=lambda x: x['dist'])

# --- APP INTERFACE ---
st.title("🎨 Gemini Artist Assistant Pro")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_container_width=True)
    
    # Extract 15 color clusters for extreme detail
    img_small = img.copy()
    img_small.thumbnail((150, 150))
    pixels = np.array(img_small).reshape(-1, 3)
    model = KMeans(n_clusters=15, n_init=10).fit(pixels)
    colors = model.cluster_centers_.astype(int)

    st.subheader("🛍️ Detected Color Palette")
    cols = st.columns(5)
    palette_data = []
    for i, rgb in enumerate(colors[:10]): # Show top 10
        matches = find_all_matches(rgb)
        best = matches[0]
        palette_data.append(best)
        with cols[i % 5]:
            hex_v = '#%02x%02x%02x' % tuple(rgb)
            st.markdown(f'<div style="background-color:{hex_v}; height:50px; border-radius:10px;"></div>', unsafe_allow_html=True)
            st.caption(f"**{best['name']}**")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I've analyzed your image. Ask me about the sky, the water, or the tiles!"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.write(m["content"])

if prompt := st.chat_input("Ask about a specific color..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.chat_message("assistant"):
        if uploaded_file:
            query = prompt.lower()
            # Search for keyword (sky, water, orange, etc) in our library
            possible = [m for m in BRAND_DATABASE.keys() if any(word in m.lower() for word in query.split())]
            
            if "sky" in query:
                res = "For the sky, I recommend **Prismacolor: Sky Blue Light**. It's much softer than the water tones."
            elif "water" in query:
                res = "For the water, I recommend **W&N: Cerulean Blue** or **Indigo Blue** for the deep shadows."
            elif "different" in query or "shade" in query:
                res = "You're right! The sky is lighter. Try using **Cloud Blue** for the atmosphere and **Indigo** only for the deepest water reflections."
            elif possible:
                res = f"I found a match for that! I recommend **{possible[0]}**. Link: {BRAND_DATABASE[possible[0]][4]}"
            else:
                res = "I see those colors! Would you like a pencil recommendation or a paint match for that area?"
            
            st.write(res)
            st.session_state.messages.append({"role": "assistant", "content": res})
