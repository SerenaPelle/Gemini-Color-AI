import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# --- BRAND DATABASE (same as before) ---
# --- EXPANDED BRAND DATABASE ---
BRAND_DATABASE = {
    # PAINTS
    "Sherwin-Williams: Naval": [47, 61, 80, "Paint", "https://www.sherwin-williams.com"],
    "Sherwin-Williams: Garden Sage": [125, 126, 100, "Paint", "https://www.sherwin-williams.com"],
    "Benjamin Moore: Hunter Green": [53, 66, 56, "Paint", "https://www.benjaminmoore.com"],
    "Farrow & Ball: Sap Green": [103, 110, 75, "Paint", "https://www.farrow-ball.com"],
    "Behr: Terra Cotta Tile": [186, 115, 87, "Paint", "https://www.behr.com"],
    "Valspar: Warm Apricot": [235, 180, 120, "Paint", "https://www.valspar.com"],
    
    # PENCILS (Prismacolor)
    "Prismacolor: Olive Green (PC911)": [115, 125, 65, "Pencil", "https://www.prismacolor.com"],
    "Prismacolor: Sand (PC940)": [220, 195, 145, "Pencil", "https://www.prismacolor.com"],
    "Prismacolor: Espresso (PC1099)": [65, 50, 45, "Pencil", "https://www.prismacolor.com"],
    "Prismacolor: Sky Blue Light (PC1086)": [180, 215, 235, "Pencil", "https://www.prismacolor.com"],
    
    # ARTIST OILS/WATERCOLORS
    "Winsor & Newton: Yellow Ochre": [195, 155, 75, "Oil Paint", "https://www.winsornewton.com"],
    "Winsor & Newton: Burnt Umber": [70, 55, 45, "Oil Paint", "https://www.winsornewton.com"],
    "Holbein: Sap Green": [80, 105, 55, "Watercolor", "https://www.holbeinartistmaterials.com"],
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

# Generate smarter "AI" response
    with st.chat_message("assistant"):
        if uploaded_file:
            response = f"I've analyzed the landscape! For those lush trees, I recommend **{found_matches[1]}**. For the stone paths, **{found_matches[2]}** is a great match. What kind of art are you planning to create with these?"
        else:
            response = "I'm ready to help! Please upload an image so I can analyze the specific shades for you."


