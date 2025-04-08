import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        # Chroma feature
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_scaled = np.mean(chroma.T, axis=0)

        # Root Mean Square Energy
        rms = librosa.feature.rms(y=audio)
        rms_scaled = np.mean(rms.T, axis=0)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_scaled = np.mean(zcr.T, axis=0)

        # Combine all features
        combined_features = np.hstack([mfccs_scaled, chroma_scaled, rms_scaled, zcr_scaled])

        return combined_features

    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}\n{e}")
        return None
