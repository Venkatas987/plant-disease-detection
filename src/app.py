import streamlit as st
from PIL import Image
import numpy as np

# --- MOCK PREDICT FUNCTION (Replace with your actual import) ---
# from inference import predict 
def predict(image):
    # This is a placeholder for your actual model logic
    return "Blueberry___healthy", 0.924

# ---------------- 1. PAGE CONFIG ----------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- 2. FULL CSS (TRANSFORMED FROM YOUR CODE) ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Variables from your index.css */
    :root {
        --background: #0b0e14;
        --card: #161b22;
        --primary: #4fd1c5;
        --border: #30363d;
        --muted: #8b949e;
        --success: #22c55e;
        --font-display: 'Space Grotesk', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* Global Styles */
    .stApp {
        background-color: var(--background);
        color: white;
        font-family: var(--font-display);
    }

    /* Header Section */
    .header-container {
        text-align: center;
        padding: 2rem 0;
    }
    .badge-pill {
        background: rgba(79, 209, 197, 0.1);
        border: 1px solid var(--primary);
        color: var(--primary);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .main-title {
        font-size: clamp(2rem, 5vw, 4rem);
        font-weight: 800;
        margin: 10px 0;
    }
    .highlight { color: var(--primary); }

    /* RESPONSIVE IMAGE CONTAINER */
    .img-wrapper {
        width: 100%;
        border-radius: 1.5rem;
        overflow: hidden;
        border: 1px solid var(--border);
        background: var(--card);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Force image to fill container width during zoom */
    .stImage img {
        width: 100% !important;
        height: auto !important;
        object-fit: contain;
        border-radius: 1rem;
    }

    /* Glass Card for Results */
    .glass-card {
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 1.5rem;
        padding: 2rem;
        height: 100%;
    }

    /* Typography for Results */
    .stat-label {
        color: var(--muted);
        text-transform: uppercase;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }

    /* Teal Progress Bar */
    .stProgress > div > div > div > div {
        background-color: var(--primary) !important;
    }

    /* Success/Warning Badges */
    .result-badge {
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        margin-top: 1rem;
    }
    .healthy { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e44; }
    .disease { background: rgba(234, 179, 8, 0.2); color: #fbbf24; border: 1px solid #eab30844; }

    /* Hide Streamlit Header */
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------- 3. RENDER UI ----------------

# Header
st.markdown("""
<div class="header-container">
    <span class="badge-pill">🧬 CNN-POWERED ANALYSIS</span>
    <h1 class="main-title">Plant Disease <span class="highlight">Detection</span></h1>
    <p style="color: #8b949e;">Identify 38+ disease classes instantly using deep learning</p>
</div>
""", unsafe_allow_html=True)

# File Uploader Container
_, col_mid, _ = st.columns([1, 4, 1])
with col_mid:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

st.markdown("---")

# ---------------- 4. LOGIC & RESPONSIVE LAYOUT ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Using 1:1 ratio columns for side-by-side display
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # Wrapper div ensures the image stays bounded and scales with zoom
        st.markdown('<div class="img-wrapper">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("↺ Upload Another Image"):
            st.rerun()

    with col2:
        st.markdown('### 📊 Prediction Result')
        
        # Inference
        with st.spinner("Analyzing Leaf Patterns..."):
            label, confidence = predict(image)
        
        clean_label = label.replace("___", " • ").replace("_", " ")

        # Disease Label
        st.markdown('<p class="stat-label">Predicted Disease</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="stat-value">{clean_label}</p>', unsafe_allow_html=True)

        # Confidence Bar
        st.markdown('<p class="stat-label">Confidence Score</p>', unsafe_allow_html=True)
        st.progress(float(confidence))
        st.markdown(f'<p style="font-family: var(--font-mono); font-weight:700;">{confidence*100:.1f}%</p>', unsafe_allow_html=True)

        # Status Badge
        if "healthy" in label.lower():
            st.markdown('<div class="result-badge healthy">✅ Plant appears healthy</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-badge disease">⚠️ Disease detected</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Empty state when no image is uploaded
    st.info("Please upload a leaf image to begin analysis.")

# ---------------- 5. FOOTER ----------------
st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #4b5563; font-size: 12px;">
    Built with CNN • Plant Disease Detection System 2026
</div>
""", unsafe_allow_html=True)