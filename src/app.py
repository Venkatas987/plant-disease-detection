import streamlit as st
from PIL import Image
from inference import predict

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.title {
    font-size: 34px;
    font-weight: 700;
    color: #4fd1c5;
}
.subtitle {
    font-size: 16px;
    color: #a0aec0;
}
.upload-box {
    max-width: 300px;
    margin: auto;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #2d3748;
}
.label {
    color: #cbd5e0;
    font-size: 14px;
}
.value {
    font-size: 18px;
    font-weight: 600;
}
.footer {
    text-align: center;
    color: #718096;
    font-size: 13px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🌿 Plant Disease Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a leaf image to identify plant diseases using a CNN model</div>',
    unsafe_allow_html=True
)

# ---------------- UPLOAD BOX (NARROWED) ----------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PROCESS ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    # -------- LEFT: IMAGE (REDUCED SIZE) --------
    with col1:
        st.image(
            image,
            width=420   # 🔥 reduced image size
        )

    # -------- RIGHT: PREDICTION --------
    with col2:
        st.markdown("#### 🔍 Prediction Result")

        with st.spinner("Analyzing image..."):
            label, confidence = predict(image)

        clean_label = label.replace("___", " | ").replace("_", " ")


        st.markdown('<div class="label">Predicted Disease</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="value">{clean_label}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="label">Confidence</div>', unsafe_allow_html=True)
        st.progress(float(confidence))
        st.markdown(f'<div class="value">{confidence*100:.2f}%</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if "healthy" in label.lower():
            st.success("✅ Plant appears healthy")
        else:
            st.warning("⚠️ Disease detected")

        st.markdown('</div>', unsafe_allow_html=True)


