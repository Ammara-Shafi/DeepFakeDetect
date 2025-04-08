import os

def count_images(folder):
    for subfolder in ['real', 'fake']:
        path = os.path.join(folder, subfolder)
        count = len([f for f in os.listdir(path) if f.endswith('.png')])
        print(f"{subfolder.title()}: {count} images")

print("🔍 Training Set:")
count_images("C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/spectrograms/train")

print("\n🧪 Test Set:")
count_images("C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/spectrograms/test")

