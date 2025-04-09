import streamlit as st
import os
from predict import predict_audio, wav_to_spectrogram
from PIL import Image

st.set_page_config(page_title="🎤 Deepfake Audio Detector")

st.title("🎙️ Deepfake Audio Detection")
st.write("Upload a `.wav` file to check if it’s **Real** or **Fake** audio.")

# File uploader
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file to disk temporarily
    file_path = os.path.join("temp.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(file_path, format="audio/wav")

    # Run prediction
    with st.spinner("Analyzing..."):
        label, confidence = predict_audio(file_path)
        spectrogram_path = wav_to_spectrogram(file_path)

    # Show prediction
    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"#### 🔍 Confidence: `{confidence * 100:.2f}%`")

    # Show spectrogram
    st.image(Image.open(spectrogram_path), caption="Spectrogram", use_container_width=True)
