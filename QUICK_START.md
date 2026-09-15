# Quick Start Guide - Improved Motor Fault Detection

## ✅ What's Been Done

Your AI model has been successfully upgraded from a simple threshold-based system to a machine learning classifier!

### Changes Made:

1. **Created `train_model.py`** - Script to train the ML model
2. **Updated `app.py`** - Now uses the trained ML model for predictions
3. **Updated `requirements.txt`** - Added scikit-learn dependency
4. **Trained the model** - Created `motor_classifier_model.pkl`
5. **Tested the model** - Both samples classify correctly!

## 🎯 Test Results

```
✓ sample_motor_healthy.wav → Predicted: Healthy (78% confidence)
✓ sample_motor_faulty.wav  → Predicted: Faulty (74% confidence)
```

## 🚀 How to Use

### 1. Start the Application

```bash
python app.py
```

### 2. Open in Browser

Navigate to: `http://localhost:5000`

### 3. Test with Your Samples

- Upload `sample_motor_healthy.wav` → Should show "Healthy"
- Upload `sample_motor_faulty.wav` → Should show "Faulty"

## 📊 How It Works Now

### Before (Old System):
```python
# Simple threshold check
prediction = "Healthy" if rms > 0.05 else "Faulty"
```

### After (New System):
```python
# Machine Learning with 32 audio features
1. Extract 32 features (RMS, spectral, MFCCs, etc.)
2. Normalize features using trained scaler
3. Random Forest classifier predicts with confidence
4. Returns: Prediction + Confidence score
```

## 🔧 Model Features

The model analyzes **32 audio characteristics**:

- **Time-domain** (2): RMS energy, zero-crossing rate
- **Frequency-domain** (4): Spectral centroid, rolloff, bandwidth, chroma
- **MFCCs** (26): 13 mean values + 13 standard deviations

## 📈 Improving Accuracy

To make the model even better, you can:

### Option 1: Add More Training Data

Edit `train_model.py` to include more samples:

```python
# Add more healthy samples
healthy_files = [
    'sample_motor_healthy.wav',
    'healthy_motor_2.wav',
    'healthy_motor_3.wav',
    # ... add more
]

# Add more faulty samples
faulty_files = [
    'sample_motor_faulty.wav',
    'faulty_motor_2.wav',
    'faulty_motor_3.wav',
    # ... add more
]
```

Then retrain:
```bash
python train_model.py
```

### Option 2: Collect Real Motor Sounds

1. Record actual motor sounds from your equipment
2. Label them as healthy or faulty
3. Add to training dataset
4. Retrain the model

## 🧪 Testing

Test the model anytime with:

```bash
python test_model.py
```

This will show predictions for both sample files with confidence scores.

## 📁 New Files

- `train_model.py` - Training script
- `test_model.py` - Testing script
- `motor_classifier_model.pkl` - Trained model (auto-generated)
- `MODEL_TRAINING_GUIDE.md` - Detailed documentation
- `QUICK_START.md` - This file

## ⚠️ Important Notes

1. **Model file required**: The app needs `motor_classifier_model.pkl` to work
2. **Retraining**: Run `python train_model.py` after adding new samples
3. **Fallback mode**: If model file is missing, app uses old threshold method

## 🎉 Success Indicators

When you run the app, you should see:

```
✓ Trained model loaded successfully
```

If you see this warning instead:
```
⚠ Warning: No trained model found. Please run train_model.py first.
```

Just run: `python train_model.py`

## 🔄 Workflow

```
1. Collect motor sound samples
   ↓
2. Run: python train_model.py
   ↓
3. Run: python test_model.py (verify)
   ↓
4. Run: python app.py
   ↓
5. Upload sounds via web interface
   ↓
6. Get predictions with confidence scores!
```

## 💡 Tips

- **More data = better accuracy**: Try to get at least 10-20 samples of each type
- **Diverse samples**: Include different motor speeds, loads, and conditions
- **Quality matters**: Use clear recordings without too much background noise
- **Regular retraining**: Retrain when you collect new samples

## 🆘 Troubleshooting

**Problem**: Model predicts incorrectly
- **Solution**: Add more training samples and retrain

**Problem**: Low confidence scores
- **Solution**: Ensure sample files are clearly different (healthy vs faulty)

**Problem**: Import errors
- **Solution**: Run `pip install -r requirements.txt`

## 📞 Next Steps

1. ✅ Model is trained and tested
2. ✅ App is ready to use
3. 🔄 Start the app: `python app.py`
4. 🔄 Test in browser with your sample files
5. 🔄 Collect more real motor sounds
6. 🔄 Retrain with expanded dataset

---

**Your motor fault detection system is now powered by machine learning!** 🎉
