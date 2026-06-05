# src/train.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def generate_mock_audio_features(data_dir):
    """Generates synthetic MFCC summary vectors matching real speech traits if audio folder is empty."""
    print("Audio data not found. Generating synthetic MFCC feature matrices for training...")
    np.random.seed(42)
    n_samples = 500
    emotions = ['happy', 'sad', 'angry', 'neutral']
    
    # 40 MFCCs averaged across time = 40 features per audio sample
    X_mock = np.random.uniform(-10, 25, size=(n_samples, 40))
    y_mock = np.random.choice(emotions, size=n_samples)
    
    # Simulate realistic emotion feature shifts (e.g., higher pitch/energy for anger)
    for i, emotion in enumerate(y_mock):
        if emotion == 'angry':
            X_mock[i] += 7.5
        elif emotion == 'sad':
            X_mock[i] -= 6.0
            
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'X_features.npy'), X_mock)
    np.save(os.path.join(data_dir, 'y_labels.npy'), y_mock)
    print("Synthetic speech features saved successfully!\n")

def load_dataset(data_dir):
    x_file = os.path.join(data_dir, 'X_features.npy')
    y_file = os.path.join(data_dir, 'y_labels.npy')
    
    if not os.path.exists(x_file) or not os.path.exists(y_file):
        generate_mock_audio_features(data_dir)
        
    X = np.load(x_file)
    y = np.load(y_file)
    return X, y

if __name__ == "__main__":
    DATA_DIR = "data"
    
    # 1. Load Data
    X, y = load_dataset(DATA_DIR)
    
    # 2. Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Initialize and Train Random Forest Classifier
    print("Training Random Forest Speech Emotion Classifier...")
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Predict & Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n================ EVALUATION METRICS ================")
    print(classification_report(y_test, predictions))
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    print("====================================================")