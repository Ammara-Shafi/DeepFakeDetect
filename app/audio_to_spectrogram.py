import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np  # Don't forget this!

# Turn off GUI backend for safe image saving
plt.switch_backend('Agg')

def create_spectrogram(file_path, output_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')  # no axis in the image
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error: {e} for {file_path}")

def process_dataset(source_dir, target_dir, label):
    source_path = os.path.join(source_dir, label)
    target_path = os.path.join(target_dir, label)

    os.makedirs(target_path, exist_ok=True)

    for file in os.listdir(source_path):
        if file.endswith(".wav"):
            input_path = os.path.join(source_path, file)
            output_path = os.path.join(target_path, file.replace(".wav", ".png"))
            create_spectrogram(input_path, output_path)

if __name__ == "__main__":
    # Train - real
    process_dataset(
        "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/data/train",
        "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/spectrograms/train",
        "real"
    )

    # Train - fake
    process_dataset(
        "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/data/train",
        "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/spectrograms/train",
        "fake"
    )

    # Test - real
    process_dataset(
        "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/data/test",
        "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/spectrograms/test",
        "real"
    )

    # Test - fake
    process_dataset(
        "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/data/test",
        "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/spectrograms/test",
        "fake"
    )

