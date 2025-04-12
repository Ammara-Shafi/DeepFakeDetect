# 🕵️ DeepFake Audio Detector 🎙️

This app helps detect whether an audio file is real or AI-generated using a CNN-based spectrogram classifier.

🔗 **Try it live**: https://deepfakedetect-idxkdvljoews8rendlv7td.streamlit.app/

-----

## 🚀 How it works

1. Upload a `.wav` audio file
2. It converts it to a mel spectrogram
3. A trained CNN model analyzes the image
4. The app outputs:
   - ✅ Prediction: *Real* or *Fake*
   - 📊 Confidence score
   - 🖼️ Visual spectrogram

-----

## 🧠 Model Info

- Model: Convolutional Neural Network (CNN)
- Input: Spectrogram images (128x128)
- Frameworks: TensorFlow, Librosa, Streamlit

-----

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Librosa
- Matplotlib
- PIL
- NumPy / Pandas

-----

## 📂 Files in Repo

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit interface |
| `predict.py` | Helper functions (preprocessing, prediction) |
| `cnn_model.h5` | Pretrained model file |
| `requirements.txt` | App dependencies |

-----


## 📩 Contact

Made by Ammara Muhammad Shafi 
💌 ammara_shafi22@hotmail.com 
🔗 [GitHub](https://github.com/Ammara-Shafi)


