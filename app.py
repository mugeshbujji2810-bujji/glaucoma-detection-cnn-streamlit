import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/glaucoma_cnn.h5"
IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)

# Page config
st.set_page_config(page_title="Glaucoma Detector", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👁️ Glaucoma Detection System")
st.write("Upload a retinal image to detect glaucoma.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

def preprocess(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing..."):
            processed = preprocess(image)
            prediction = model.predict(processed)[0][0]

            if prediction > 0.5:
                st.error(f"⚠️ Glaucoma Detected ({prediction*100:.2f}%)")
            else:
                st.success(f"✅ Normal Eye ({(1-prediction)*100:.2f}%)")

        st.progress(int(prediction * 100))