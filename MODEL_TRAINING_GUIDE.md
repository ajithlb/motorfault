# Motor Fault Detection - Model Training Guide

## Overview
This guide explains how to train and use the improved AI model for motor fault detection.

## What Changed?

### Before:
- Simple threshold-based prediction (RMS > 0.05 = Healthy)
- Not learning from actual motor sounds
- Low accuracy

### After:
- Machine Learning model (Random Forest Classifier)
- Learns patterns from your sample audio files
- Extracts 32 audio features including:
  - Time-domain: RMS energy, zero-crossing rate
  - Frequency-domain: spectral centroid, rolloff, bandwidth
  - MFCCs: 13 coefficients (mean and std)
  - Chroma features
- Provides confidence scores

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script to create the ML model from your sample files:

```bash
python train_model.py
```

This will:
- Load `sample_motor_healthy.wav` and `sample_motor_faulty.wav`
- Extract audio features from both samples
- Train a Random Forest classifier
- Save the trained model to `motor_classifier_model.pkl`

**Expected Output:**
```
Starting model training...
Processing sample_motor_healthy.wav...
  ✓ Extracted 32 features
Processing sample_motor_faulty.wav...
  ✓ Extracted 32 features

Dataset: 2 samples, 32 features per sample
  Healthy samples: 1
  Faulty samples: 1

Training Random Forest classifier...

Training accuracy: 100.00%

Classification Report:
              precision    recall  f1-score   support

     Healthy       1.00      1.00      1.00         1
      Faulty       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

✓ Model saved to 'motor_classifier_model.pkl'

Model training complete!
```

### 3. Run the Application
```bash
python app.py
```

The app will automatically load the trained model on startup.

## How It Works

### Training Phase (train_model.py):
1. Loads sample audio files
2. Extracts 32 audio features from each sample
3. Trains a Random Forest classifier with 100 decision trees
4. Saves the model and feature scaler

### Prediction Phase (app.py):
1. User uploads an audio file
2. System extracts the same 32 features
3. Features are normalized using the saved scaler
4. Model predicts: Healthy (0) or Faulty (1)
5. Returns prediction with confidence score

## Improving Model Accuracy

To improve the model with more training data:

### Option 1: Add More Sample Files
1. Modify `train_model.py` to load additional files
2. Add loops to process multiple healthy/faulty samples
3. Retrain the model

### Option 2: Data Augmentation
Add variations of existing samples:
- Time stretching
- Pitch shifting
- Adding noise
- Speed changes

### Option 3: Collect Real Data
- Record actual motor sounds in your environment
- Label them as healthy or faulty
- Add to training dataset

## Testing the Model

Test with your sample files:

1. Start the application: `python app.py`
2. Open browser: `http://localhost:5000`
3. Upload `sample_motor_healthy.wav` → Should predict "Healthy"
4. Upload `sample_motor_faulty.wav` → Should predict "Faulty"

## Troubleshooting

### Model Not Found Warning
```
⚠ Warning: No trained model found. Please run train_model.py first.
```
**Solution:** Run `python train_model.py` to create the model file.

### Low Accuracy
If the model doesn't predict correctly:
1. Check that sample files are different enough
2. Add more training samples
3. Verify audio quality (not corrupted)
4. Try adjusting feature extraction parameters

### Import Errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution:** Install dependencies: `pip install -r requirements.txt`

## Model Files

- `motor_classifier_model.pkl` - Trained model (created by train_model.py)
- `train_model.py` - Training script
- `app.py` - Web application with prediction

## Next Steps

1. ✅ Train the model with your samples
2. ✅ Test predictions in the web interface
3. 🔄 Collect more real motor sounds
4. 🔄 Retrain with expanded dataset
5. 🔄 Deploy to production

## Technical Details

**Model:** Random Forest Classifier
- 100 estimators (decision trees)
- Max depth: 10
- Handles small datasets well

**Features:** 32 total
- 6 basic audio features
- 13 MFCC means
- 13 MFCC standard deviations

**Preprocessing:** StandardScaler normalization
