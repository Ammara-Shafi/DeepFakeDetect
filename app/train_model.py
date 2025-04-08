import os   
import numpy as np  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from extract_features import extract_features

#-------------------------------------------------------------------------
#defining datsets
train_real_dir = "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/data/train/real"
train_fake_dir = "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/data/train/fake"

test_real_dir =  "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/data/test/real"
test_fake_dir =  "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/data/test/fake"

#-------------------------------------------------------------------------

def load_data(real_dir, fake_dir):
    features = []
    labels = []

    #load real audio files
    print(f"Loading real audion from {real_dir}")
    for file in os.listdir(real_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(real_dir, file)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(0) #real

    #load fake audio files
    print(f"Loading fake audio from {fake_dir}")
    for file in os.listdir(fake_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(fake_dir, file)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(1)  #fake

    return np.array(features), np.array(labels)


# Step 1: Load training data
X_train, y_train = load_data(train_real_dir, train_fake_dir)
print("🔍 Checking training dataset balance:")
print(f"Real samples (0): {np.sum(y_train == 0)}")
print(f"Fake samples (1): {np.sum(y_train == 1)}")


# Step 2: Train the model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Load test data
X_test, y_test = load_data(test_real_dir, test_fake_dir)
print("🔍 Checking test dataset balance:")
print(f"Real samples (0): {np.sum(y_test == 0)}")
print(f"Fake samples (1): {np.sum(y_test == 1)}")

# Step 4: Evaluate model
print("Evaluating on test set...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 5: Save the model
model_path = "C:/Users/amsh/OneDrive - Boskalis/Desktop/deepfake_audio_detector/model/audio_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")




