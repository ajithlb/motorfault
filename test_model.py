"""
Test the trained model with sample files to verify it works correctly.
"""

import os
import pickle
import librosa
import numpy as np

def extract_ml_features(filepath):
    """Extract features from audio file."""
    y, sr = librosa.load(filepath, sr=22050)
    
    # Time-domain features
    rms = librosa.feature.rms(y=y).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    
    # Frequency-domain features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    
    # MFCCs
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

def test_model():
    """Test the trained model with sample files."""
    print("=" * 60)
    print("Testing Motor Fault Detection Model")
    print("=" * 60)
    
    # Load the model
    try:
        with open('motor_classifier_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print("\n✓ Model loaded successfully\n")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}\n")
        return
    
    # Test files
    test_files = [
        ('sample_motor_healthy.wav', 'Healthy'),
        ('sample_motor_faulty.wav', 'Faulty')
    ]
    
    print("Testing predictions:\n")
    print("-" * 60)
    
    for filepath, expected in test_files:
        if not os.path.exists(filepath):
            print(f"✗ {filepath} not found")
            continue
        
        try:
            # Extract features
            features = extract_ml_features(filepath)
            
            # Scale features
            features_scaled = model_data['scaler'].transform([features])
            
            # Predict
            prediction_label = model_data['classifier'].predict(features_scaled)[0]
            prediction_proba = model_data['classifier'].predict_proba(features_scaled)[0]
            
            prediction = "Healthy" if prediction_label == 0 else "Faulty"
            confidence = prediction_proba[prediction_label] * 100
            
            # Check if correct
            status = "✓" if prediction == expected else "✗"
            
            print(f"{status} File: {filepath}")
            print(f"  Expected:   {expected}")
            print(f"  Predicted:  {prediction}")
            print(f"  Confidence: {confidence:.1f}%")
            print(f"  Probabilities: Healthy={prediction_proba[0]*100:.1f}%, Faulty={prediction_proba[1]*100:.1f}%")
            print("-" * 60)
            
        except Exception as e:
            print(f"✗ Error testing {filepath}: {e}")
            print("-" * 60)
    
    print("\nTest complete!")
    print("=" * 60)

if __name__ == '__main__':
    test_model()
