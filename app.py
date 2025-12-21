import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# --- UI CONFIGURATION (Gemini Style) ---
st.set_page_config(page_title="Gemini Color AI", page_icon="✨", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #1a73e8; color: white; border-radius: 20px; }
    .color-card { padding: 20px; border-radius: 15px; margin: 10px; text-align: center; color: white; font-weight: bold; text-shadow: 1px 1px 2px #000; }
    </style>
    """, unsafe_allow_html=True)

st.title("✨ Gemini Color Analysis AI")
st.write("Upload an image, and I will analyze the aesthetic DNA and dominant color palettes for you.")

# --- HELPER FUNCTIONS ---
def get_color_name(rgb):
    try:
        return webcolors.rgb_to_name(rgb)
    except:
        return "Custom Shade"

def analyze_colors(image, num_colors=5):
    # Resize to speed up AI processing
    img = image.copy()
    img.thumbnail((200, 200))
    img_array = np.array(img.convert('RGB'))
    
    # Flatten the image into a list of pixels
    pixels = img_array.reshape(-1, 3)
    
    # Use K-Means AI to find the 'clusters' of colors
    model = KMeans(n_clusters=num_colors, n_init=10)
    model.fit(pixels)
    
    return model.cluster_centers_.astype(int)

# --- APP LOGIC ---
uploaded_file = st.file_uploader("Drop an image here...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("🤖 AI Analysis")
        with st.spinner("Analyzing spectral data..."):
            colors = analyze_colors(image)
            
            st.write("I've identified these as the dominant tones in your image:")
            
            for color in colors:
                hex_val = '#%02x%02x%02x' % tuple(color)
                st.markdown(f"""
                    <div class="color-card" style="background-color: {hex_val};">
                        {hex_val.upper()}
                    </div>
                """, unsafe_allow_html=True)
                
    st.success("Analysis complete. These colors represent the core visual identity of your upload.")
