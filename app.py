import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# --- BRAND DATABASE (same as before) ---
# --- PROFESSIONAL ARTIST DATABASE ---
BRAND_DATABASE = {
    # SKY & WATER BLUES
    "Prismacolor: Sky Blue Light (PC1086)": [180, 215, 235, "Pencil", "https://www.prismacolor.com"],
    "Prismacolor: Cloud Blue (PC1023)": [190, 210, 225, "Pencil", "https://www.prismacolor.com"],
    "Winsor & Newton: Cerulean Blue": [42, 82, 190, "Oil Paint", "https://www.winsornewton.com"],
    "Sherwin-Williams: Sky High": [210, 225, 235, "Paint", "https://www.sherwin-williams.com"],
    "Caran d'Ache: Ultramarine": [18, 10, 143, "Pencil", "https://www.carandache.com"],

    # FOLIAGE & LANDSCAPE GREENS
    "Prismacolor: Olive Green (PC911)": [115, 125, 65, "Pencil", "https://www.prismacolor.com"],
    "Prismacolor: Apple Green (PC912)": [140, 185, 70, "Pencil", "https://www.prismacolor.com"],
    "Holbein: Sap Green": [80, 105, 55, "Watercolor", "https://www.holbeinartistmaterials.com"],
    "Sherwin-Williams: Garden Sage": [125, 126, 100, "Paint", "https://www.sherwin-williams.com"],

    # EARTH, STONE & HIGHLIGHTS
    "Winsor & Newton: Yellow Ochre": [195, 155, 75, "Oil Paint", "https://www.winsornewton.com"],
    "Winsor & Newton: Burnt Umber": [70, 55, 45, "Oil Paint", "https://www.winsornewton.com"],
    "Prismacolor: Sand (PC940)": [220, 195, 145, "Pencil", "https://www.prismacolor.com"],
    "Farrow & Ball: Old White": [215, 210, 195, "Paint", "https://www.farrow-ball.com"],
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

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Current Reference", use_container_width=True)
    
    # --- STEP 1: INCREASE VISION TO 10 COLORS ---
    img_small = image.copy().convert('RGB')
    img_small.thumbnail((100, 100))
    pixels = np.array(img_small).reshape(-1, 3)
    
    # We now look for 10 colors to ensure we catch the sky/details
    model = KMeans(n_clusters=10, n_init=10).fit(pixels)
    colors = model.cluster_centers_.astype(int)

    st.write("### 🛍️ Top Matches Found")
    cols = st.columns(5) # Show top 5 visually
    found_matches = []
    
    for i, rgb in enumerate(colors):
        product_name, info = find_closest_product(rgb)
        found_matches.append({"name": product_name, "rgb": rgb, "info": info})
        
        # Display only the first 5 in the UI to keep it clean
        if i < 5:
            hex_val = '#%02x%02x%02x' % tuple(rgb)
            with cols[i]:
                st.markdown(f'<div style="background-color:{hex_val}; height:40px; border-radius:10px;"></div>', unsafe_allow_html=True)
                st.caption(f"**{product_name}**")

# --- CHAT INTERFACE ---
st.divider()
st.subheader("💬 Ask Your Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: 'Which blue is best for a sky?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if uploaded_file:
            user_query = prompt.lower()
            
            # --- STEP 2: SMART SEARCH LOGIC ---
            # If user asks for blue, search ALL 10 found colors for a 'Blue' or 'Sky' product
            if "blue" in user_query or "sky" in user_query:
                blue_matches = [m["name"] for m in found_matches if "blue" in m["name"].lower() or "sky" in m["name"].lower()]
                
                if blue_matches:
                    response = f"I found a great sky match! Based on the blues in your photo, I recommend **{blue_matches[0]}**."
                else:
                    response = "I can see the sky, but it's quite pale. I'd recommend using **Prismacolor: Sky Blue Light** for those areas."
            else:
                response = f"For the main parts of this image, I recommend **{found_matches[0]['name']}**."
        else:
            response = "Please upload an image so I can analyze the colors!"
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
