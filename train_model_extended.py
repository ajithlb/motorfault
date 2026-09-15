"""
Train a machine learning model using the extended dataset.
Uses all generated motor sounds (10 healthy + 10 faulty) plus original samples.
"""

import os
import numpy as np
import librosa
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob

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

def load_dataset():
    """
    Load all audio files from the training dataset.
    """
    X = []  # Features
    y = []  # Labels (0 = Healthy, 1 = Faulty)
    filenames = []
    
    print("\n📂 Loading dataset...")
    
    # Load original samples
    original_samples = [
        ('sample_motor_healthy.wav', 0, 'Healthy'),
        ('sample_motor_faulty.wav', 1, 'Faulty')
    ]
    
    for filepath, label, label_name in original_samples:
        if os.path.exists(filepath):
            print(f"  Loading: {filepath} ({label_name})")
            features = extract_features(filepath)
            if features is not None:
                X.append(features)
                y.append(label)
                filenames.append(filepath)
    
    # Load generated healthy samples
    healthy_files = glob.glob('training_data/healthy/*.wav')
    print(f"\n  Found {len(healthy_files)} generated healthy samples")
    for filepath in sorted(healthy_files):
        features = extract_features(filepath)
        if features is not None:
            X.append(features)
            y.append(0)  # 0 = Healthy
            filenames.append(filepath)
            print(f"    ✓ {os.path.basename(filepath)}")
    
    # Load generated faulty samples
    faulty_files = glob.glob('training_data/faulty/*.wav')
    print(f"\n  Found {len(faulty_files)} generated faulty samples")
    for filepath in sorted(faulty_files):
        features = extract_features(filepath)
        if features is not None:
            X.append(features)
            y.append(1)  # 1 = Faulty
            filenames.append(filepath)
            print(f"    ✓ {os.path.basename(filepath)}")
    
    return np.array(X), np.array(y), filenames

def train_motor_classifier():
    """
    Train a classifier using the extended motor sound dataset.
    """
    print("=" * 70)
    print("Training Motor Classifier with Extended Dataset")
    print("=" * 70)
    
    # Load dataset
    X, y, filenames = load_dataset()
    
    # Check if we have enough data
    if len(X) < 4:
        print("\n✗ Error: Need at least 4 samples to train the model!")
        print("  Please run: python generate_motor_sounds.py")
        return False
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Healthy samples: {np.sum(y == 0)}")
    print(f"  Faulty samples: {np.sum(y == 1)}")
    print(f"  Features per sample: {X.shape[1]}")
    
    # Normalize features
    print("\n🔧 Preprocessing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Train classifier
    print("\n🤖 Training Random Forest classifier...")
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    classifier.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Evaluate on test set
    y_test_pred = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n📈 Model Performance:")
    print(f"  Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"  Test accuracy: {test_accuracy * 100:.2f}%")
    
    # Cross-validation score
    if len(X) >= 5:
        cv_scores = cross_val_score(classifier, X_scaled, y, cv=min(5, len(X)), scoring='accuracy')
        print(f"  Cross-validation accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    
    # Detailed classification report
    print("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Healthy', 'Faulty'], zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("🔍 Confusion Matrix (Test Set):")
    print(f"                Predicted")
    print(f"              Healthy  Faulty")
    print(f"Actual Healthy    {cm[0][0]}       {cm[0][1]}")
    print(f"       Faulty     {cm[1][0]}       {cm[1][1]}")
    
    # Feature importance
    feature_names = [
        'rms', 'zero_crossing_rate', 'spectral_centroid', 'spectral_rolloff',
        'spectral_bandwidth', 'chroma_mean'
    ] + [f'mfcc_{i}_mean' for i in range(13)] + [f'mfcc_{i}_std' for i in range(13)]
    
    importances = classifier.feature_importances_
    top_features_idx = np.argsort(importances)[-5:][::-1]
    
    print("\n🎯 Top 5 Most Important Features:")
    for idx in top_features_idx:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save the model
    model_data = {
        'classifier': classifier,
        'scaler': scaler,
        'feature_names': feature_names,
        'training_samples': len(X),
        'test_accuracy': test_accuracy
    }
    
    with open('motor_classifier_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\n✅ Model saved to 'motor_classifier_model.pkl'")
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    print("\n🎯 Next Steps:")
    print("  1. Test the model: python test_model.py")
    print("  2. Test the API: python test_api_simple.py")
    print("  3. Restart Flask app: python app.py")
    
    return True

if __name__ == '__main__':
    success = train_motor_classifier()
    if not success:
        print("\n✗ Training failed. Please check the error messages above.")
        exit(1)
