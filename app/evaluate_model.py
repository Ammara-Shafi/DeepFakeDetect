from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image_dataset_from_directory #type: ignore

# Load test data
test_data = image_dataset_from_directory(
    "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/spectrograms/validation",
    label_mode='binary',
    image_size=(128, 128),
    batch_size=32,
    shuffle=False  # Don't shuffle for evaluation
)

# Path to the best model we saved
model_path = "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/model/cnn_model.h5"

# Load the trained model
model = load_model(model_path)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_data)

print(f"📊 Test Accuracy: {test_accuracy:.4f}")
print(f"🧪 Test Loss: {test_loss:.4f}")
