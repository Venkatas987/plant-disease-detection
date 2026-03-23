import streamlit as st
from PIL import Image
import base64

# --- Try to import your real predict function ---
try:
    from inference import predict
except ImportError:
    # Fallback placeholder for testing
    def predict(img): return "Potato___Early_blight", 0.88

# --- HELPER: Convert local image to base64 for the background ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

# ---------------- 1. PAGE CONFIG ----------------
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌿",
    layout="wide"
)

# Load background image
hero_base64 = get_base64_image("hero-leaf.jpg")
bg_css = f"url(data:image/jpeg;base64,{hero_base64})" if hero_base64 else "none"

# ---------------- 2. CSS (FIXED BRACES) ----------------
# We use a regular string here (no 'f') to avoid curly brace syntax errors,
# and then inject the background image separately.
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --background: #0b0e14;
        --card: #161b22;
        --primary: #4fd1c5;
        --border: #30363d;
        --muted: #8b949e;
    }}

    .stApp {{
        background-color: var(--background);
        color: white;
        font-family: 'Space Grotesk', sans-serif;
    }}

    /* Hero Background */
    .hero-section {{
        position: absolute;
        top: 0; left: 0; width: 100%; height: 400px;
        background-image: linear-gradient(to bottom, rgba(11,14,20,0.3), #0b0e14), {bg_css};
        background-size: cover; background-position: center;
        z-index: -1; opacity: 0.6;
    }}

    /* Typography */
    .title-container {{ text-align: center; padding: 5rem 0 2rem 0; }}
    .main-title {{ font-size: 4rem; font-weight: 800; margin: 0; }}
    .highlight {{ color: var(--primary); }}
    .sub-text {{ color: var(--muted); font-size: 1.1rem; margin-top: 10px; }}

    /* Cards */
    .glass-card {{
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 1.5rem;
        padding: 2.5rem;
        box-shadow: 0 0 40px rgba(79, 209, 197, 0.05);
    }}

    .stat-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.12em; }}
    .stat-value {{ font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 20px; }}
    .conf-text {{ font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; }}

    /* Progress Bar */
    .stProgress > div > div > div > div {{ background-color: var(--primary) !important; }}

    header, footer {{ visibility: hidden; }}
</style>
<div class="hero-section"></div>
""", unsafe_allow_html=True)

# ---------------- 3. HEADER ----------------
st.markdown("""
<div class="title-container">
    <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
        <span style="border: 1px solid #4fd1c5; color: #4fd1c5; padding: 4px 16px; border-radius: 20px; font-size: 12px; font-weight: 600;">
            🧬 CNN-POWERED ANALYSIS
        </span>
    </div>
    <h1 class="main-title">Plant Disease <span class="highlight">Detection</span></h1>
    <p class="sub-text">Identify diseases instantly using deep learning. Fast, accurate, and beautifully simple.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- 4. UPLOADER ----------------
_, col_up, _ = st.columns([1, 2, 1])
with col_up:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# ---------------- 5. RESULTS ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.spinner("Analyzing image..."):
            label, confidence = predict(image)
        
        clean_label = label.replace("___", " • ").replace("_", " ")

        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; color: #4fd1c5;">
                <span style="font-size: 1.5rem;">📊</span> <h3 style="margin:0;">Prediction Result</h3>
            </div>
            <p class="stat-label">Predicted Disease</p>
            <div class="stat-value">{clean_label}</div>
            <p class="stat-label">Confidence Score</p>
        """, unsafe_allow_html=True)
        
        st.progress(float(confidence))
        
        st.markdown(f"""
            <div class="conf-text">{confidence*100:.1f}%</div>
        """, unsafe_allow_html=True)

        # Logic check for healthy vs disease
        if "healthy" in label.lower():
            st.success("✅ Plant appears healthy")
        else:
            st.warning("⚠️ Disease detected")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Reset Link
    st.markdown('<div style="text-align:center; margin-top:30px;"><a href="/" target="_self" style="color:#4fd1c5; text-decoration:none; font-weight:600;">← Upload another image</a></div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; margin-top:80px; color:#4b5563; font-size:12px;">
    Built with CNN • Plant Disease Detection System 2026
</div>
""", unsafe_allow_html=True)