"""
Train a machine learning model to classify motor sounds as healthy or faulty.
This script extracts features from sample audio files and trains a classifier.
"""

import os
import numpy as np
import librosa
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def extract_features(filepath):
    """
    Extract comprehensive audio features from a file.
    """
    try:
        # Load audio file
        y, sr = librosa.load(filepath, sr=22050)
        
        # Time-domain features
        rms = librosa.feature.rms(y=y).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        
        # Frequency-domain features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        
        # MFCCs (Mel-frequency cepstral coefficients) - powerful for audio classification
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfccs.mean(axis=1)
        mfcc_std = mfccs.std(axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean()
        
        # Combine all features
        features = np.concatenate([
            [rms, zero_crossing_rate, spectral_centroid, spectral_rolloff, 
             spectral_bandwidth, chroma_mean],
            mfcc_mean,
            mfcc_std
        ])
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {filepath}: {e}")
        return None

def train_motor_classifier():
    """
    Train a classifier using sample motor sounds.
    """
    print("Starting model training...")
    
    # Prepare data
    X = []  # Features
    y = []  # Labels (0 = Healthy, 1 = Faulty)
    
    # Load healthy sample
    healthy_file = 'sample_motor_healthy.wav'
    if os.path.exists(healthy_file):
        print(f"Processing {healthy_file}...")
        features = extract_features(healthy_file)
        if features is not None:
            X.append(features)
            y.append(0)  # 0 = Healthy
            print(f"  ✓ Extracted {len(features)} features")
    else:
        print(f"Warning: {healthy_file} not found!")
    
    # Load faulty sample
    faulty_file = 'sample_motor_faulty.wav'
    if os.path.exists(faulty_file):
        print(f"Processing {faulty_file}...")
        features = extract_features(faulty_file)
        if features is not None:
            X.append(features)
            y.append(1)  # 1 = Faulty
            print(f"  ✓ Extracted {len(features)} features")
    else:
        print(f"Warning: {faulty_file} not found!")
    
    # Check if we have enough data
    if len(X) < 2:
        print("Error: Need at least 2 samples (1 healthy, 1 faulty) to train the model!")
        return False
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features per sample")
    print(f"  Healthy samples: {np.sum(y == 0)}")
    print(f"  Faulty samples: {np.sum(y == 1)}")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train classifier
    # Using RandomForest with specific parameters for small datasets
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    print("\nTraining Random Forest classifier...")
    classifier.fit(X_scaled, y)
    
    # Evaluate on training data (since we have limited samples)
    y_pred = classifier.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\nTraining accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Healthy', 'Faulty'], zero_division=0))
    
    # Save the model and scaler
    model_data = {
        'classifier': classifier,
        'scaler': scaler,
        'feature_names': [
            'rms', 'zero_crossing_rate', 'spectral_centroid', 'spectral_rolloff',
            'spectral_bandwidth', 'chroma_mean'
        ] + [f'mfcc_{i}_mean' for i in range(13)] + [f'mfcc_{i}_std' for i in range(13)]
    }
    
    with open('motor_classifier_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\n✓ Model saved to 'motor_classifier_model.pkl'")
    print("\nModel training complete!")
    
    return True

if __name__ == '__main__':
    success = train_motor_classifier()
    if not success:
        print("\nTraining failed. Please ensure you have the sample audio files.")
        exit(1)
