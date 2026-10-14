import streamlit as st
from PIL import Image
from src.inference import load_assets, predict_top_k

st.set_page_config(page_title="Emoji Classifier", page_icon="😀", layout="centered")
st.title("Image-Based Emoji Classifier")
st.write("Upload an emoji image to classify it with the trained CNN model.")

try:
    model, labels = load_assets()
    st.caption(f"Loaded model with {len(labels):,} emoji classes.")
except Exception as exc:
    st.error(f"Could not load model assets: {exc}")
    st.stop()

uploaded = st.file_uploader("Upload an emoji image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=220)

    predictions = predict_top_k(model, labels, image, k=5)
    best = predictions[0]

    st.subheader(f"Prediction: {best['label']}")
    st.metric("Model confidence", f"{best['confidence']:.2%}")

    st.subheader("Top predictions")
    for item in predictions:
        st.write(f"**{item['label']}** - {item['confidence']:.2%}")

st.caption("Portfolio/educational image-classification project.")
