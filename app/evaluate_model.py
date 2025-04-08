from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Paths
test_dir = "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/spectrograms/test"
model_path = "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/model/cnn_model.h5"

# Load model
model = load_model(model_path)

# Prepare test data (only rescale, no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Don't shuffle to keep results aligned
)

# Evaluate
loss, accuracy = model.evaluate(test_data)
print(f"\n📊 Test Accuracy: {accuracy:.4f}")
print(f"🧪 Test Loss: {loss:.4f}")
