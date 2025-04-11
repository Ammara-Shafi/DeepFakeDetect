#this will contain helper functions for my UI.

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import load_model #type: ignore
from PIL import Image

# === Load your model ===
model_path = os.path.join(os.path.dirname(__file__), "cnn_model.h5")
model = load_model(model_path)

# === Convert .wav to spectrogram and save ===
def wav_to_spectrogram(wav_path, save_path="temp_spectrogram.png"):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Plot and save as image
    plt.figure(figsize=(1.28, 1.28), dpi=100)
    librosa.display.specshow(S_DB, sr=sr, cmap='inferno')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return save_path

# === Preprocess image for model ===
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# === Make prediction ===
def predict_audio(wav_path):
    spectrogram_path = wav_to_spectrogram(wav_path)
    img_array = preprocess_image(spectrogram_path)
    prediction = model.predict(img_array)[0][0]

    label = "Real" if prediction >= 0.5 else "Fake"
    confidence = prediction if label == "Real" else 1 - prediction
    
    return label, confidence, spectrogram_path

# === 🔁 Batch prediction support ===
def predict_multiple_audios(file_paths, save_csv=False):
    results = []
    
    for file_path in file_paths:
        label, confidence, spec_path = predict_audio(file_path)
        results.append({
            "Filename": os.path.basename(file_path),
            "Prediction": label,
            "Confidence": confidence,
            "Spectrogram": spec_path  # 👈 make sure to include this
        })
        df = pd.DataFrame(results)

    if save_csv:
        csv_path = os.path.join("temp_results.csv")
        df.to_csv(csv_path, index=False)
        return df, csv_path
    
    return df, None

