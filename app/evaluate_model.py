#This evaluates my model on validation data, and also generates the confusion matrix
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image_dataset_from_directory #type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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


# Get true labels and predictions
y_true = np.concatenate([y for x, y in test_data], axis=0)
y_pred_probs = model.predict(test_data)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Print Report
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

