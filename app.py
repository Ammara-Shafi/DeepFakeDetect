import streamlit as st
import os
import pandas as pd
from predict import predict_audio, predict_multiple_audios
import tempfile
from PIL import Image

st.set_page_config(page_title="🎤 Deepfake Audio Detector", layout="centered")

st.title("🎙️ DeepfakeDetect")
st.markdown("Upload one or more `.wav` files to check if they're **Real** or **Fake** audio.")

# File uploader
uploaded_files = st.file_uploader("Choose .wav file(s)", type="wav", accept_multiple_files=True)

# === Prediction on Upload ===
if uploaded_files:
    temp_files = []

    st.subheader("🎧 Analyzing your audio file(s)...")

    # Save uploaded files temporarily
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            temp_files.append(tmp.name)

    # === SINGLE FILE VIEW (detailed) ===
    if len(temp_files) == 1:
        st.markdown(f"**🎧 File:** {uploaded_files[0].name}")
        st.audio(temp_files[0], format="audio/wav")

        label, confidence, spec_path = predict_audio(temp_files[0])

        emoji = "✅" if label == "Real" else "😵‍💫"
        conf_color = "green" if confidence > 0.5 else "red"

        st.markdown(f"### {emoji} <b>Prediction:</b> <span style='color:#444'>{label}</span>", unsafe_allow_html=True)
        st.markdown(f"### 🔍 <b>Confidence:</b> <span style='color:{conf_color}'>{confidence*100:.2f}%</span>", unsafe_allow_html=True)

        st.image(spec_path, caption="Spectrogram", use_container_width=True)

    # === MULTI FILE BATCH MODE ===
    elif len(temp_files) > 1:
        st.subheader("📦 Batch Predictions")
        results_df, csv_path = predict_multiple_audios(temp_files, save_csv=True)

        st.write("📊 Columns in results_df:", results_df.columns.tolist())

        for i, file in enumerate(uploaded_files):
            st.markdown("---")
            st.markdown(f"### 🎧 File {i+1}: **{file.name}**")

            st.audio(temp_files[i], format="audio/wav")

            label = results_df.loc[i, "Prediction"]
            confidence = results_df.loc[i, "Confidence"]
            spec_path = results_df.loc[i, "Spectrogram"]

            emoji = "✅" if label == "Real" else "😵‍💫"
            conf_color = "green" if confidence > 0.5 else "red"

            st.markdown(f"**{emoji} Prediction:** <span style='color:#444'>{label}</span>", unsafe_allow_html=True)
            st.markdown(f"**🔍 Confidence:** <span style='color:{conf_color}'>{confidence*100:.2f}%</span>", unsafe_allow_html=True)

            st.image(spec_path, caption="Spectrogram", use_container_width=True)

        # Show Dataframe + CSV Download
        st.markdown("---")
        st.subheader("📋 Full Results Table")
        st.dataframe(results_df)

        with open(csv_path, "rb") as f:
            st.download_button(
                label="⬇️ Download CSV Results",
                data=f,
                file_name="deepfake_predictions.csv",
                mime="text/csv"
            )

        # Cleanup CSV file
        if os.path.exists(csv_path):
            os.remove(csv_path)

    # Cleanup temp audio files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# 👣 Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.9em;'>✨ Created & Designed with 💜 by <strong>Ammara Muhammad Shafi</strong></p>",
    unsafe_allow_html=True
)
