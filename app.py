import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# --- BRAND DATABASE ---
# Format: "Product Name": [R, G, B, "Category", "Buy Link"]
BRAND_DATABASE = {
    "Sherwin-Williams: Naval": [47, 61, 80, "Paint", "https://www.sherwin-williams.com"],
    "Benjamin Moore: Hale Navy": [54, 65, 77, "Paint", "https://www.benjaminmoore.com"],
    "Prismacolor: Indigo Blue (PC901)": [30, 45, 80, "Pencil", "https://www.prismacolor.com"],
    "Winsor & Newton: French Ultramarine": [18, 52, 144, "Oil Paint", "https://www.winsornewton.com"],
    "Prismacolor: Sunburst Yellow (PC917)": [255, 215, 0, "Pencil", "https://www.prismacolor.com"],
    "Sherwin-Williams: Tricorn Black": [47, 47, 48, "Paint", "https://www.sherwin-williams.com"],
    "Faber-Castell: Emerald Green": [0, 150, 100, "Pencil", "https://www.fabercastell.com"],
    # You can add hundreds of these rows!
}

st.set_page_config(page_title="Gemini Art Assistant", page_icon="🎨", layout="wide")

# --- AI MATCHING ENGINE ---
def find_closest_product(target_rgb):
    closest_name = None
    min_dist = float('inf')
    
    for name, data in BRAND_DATABASE.items():
        # The Math: Distance = sqrt((R1-R2)^2 + (G1-G2)^2 + (B1-B2)^2)
        dist = np.sqrt(sum((np.array(target_rgb) - np.array(data[:3]))**2))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
            
    return closest_name, BRAND_DATABASE[closest_name]

# --- UI LAYOUT ---
st.title("🎨 Gemini Art & Paint Matcher")
st.write("Upload an image to find matching real-world paints and pencils.")

uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=400)
    
    # Process image for colors
    img_small = image.copy()
    img_small.thumbnail((100, 100))
    pixels = np.array(img_small.convert('RGB')).reshape(-1, 3)
    
    # Get 3 dominant colors
    model = KMeans(n_clusters=3, n_init=10).fit(pixels)
    colors = model.cluster_centers_.astype(int)

    st.subheader("🛍️ Recommended Supplies")
    cols = st.columns(3)

    for i, rgb in enumerate(colors):
        hex_val = '#%02x%02x%02x' % tuple(rgb)
        product_name, info = find_closest_product(rgb)
        
        with cols[i]:
            st.markdown(f'<div style="background-color:{hex_val}; height:50px; border-radius:10px;"></div>', unsafe_allow_True=True)
            st.write(f"**Target:** {hex_val}")
            st.info(f"**Match:** {product_name}\n\n*Type: {info[3]}*")
            st.link_button("View Product", info[4])
