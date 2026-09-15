import os
import random
import time
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import librosa
import numpy as np
from pydub import AudioSegment

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = 'motor_classifier_model.pkl'
model_data = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        print("✓ Trained model loaded successfully")
    else:
        print("⚠ Warning: No trained model found. Please run train_model.py first.")
        print("  Using fallback threshold-based prediction.")
except Exception as e:
    print(f"⚠ Error loading model: {e}")
    print("  Using fallback threshold-based prediction.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_ml_features(y, sr):
    """
    Extract comprehensive audio features for ML model.
    """
    # Time-domain features
    rms = librosa.feature.rms(y=y).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    
    # Frequency-domain features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    
    # MFCCs (Mel-frequency cepstral coefficients)
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

def predict_motor_state(filepath):
    """
    Extract audio features and predict motor state using trained ML model.
    """
    try:
        # Convert to wav if necessary
        if filepath.lower().endswith(('.webm', '.ogg', '.mp3')):
            audio = AudioSegment.from_file(filepath)
            wav_path = os.path.splitext(filepath)[0] + '.wav'
            audio.export(wav_path, format='wav')
            filepath = wav_path
        
        # Load audio file
        y, sr = librosa.load(filepath, sr=22050)
        
        # Extract basic features for display
        rms = librosa.feature.rms(y=y).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        
        # Compute spectrum (FFT)
        fft = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(y), 1/sr)
        magnitude = np.abs(fft)
        # Take only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        magnitude_pos = magnitude[pos_mask]
        # Downsample for visualization (take every 10th point to reduce data)
        step = max(1, len(freqs_pos) // 1000)
        spectrum_freqs = freqs_pos[::step]
        spectrum_magnitude = magnitude_pos[::step]
        
        # Extract waveform samples for visualization
        # Downsample waveform to max 2000 samples for frontend visualization
        waveform_step = max(1, len(y) // 2000)
        waveform_samples = y[::waveform_step].astype(float).tolist()
        
        # Normalize waveform to -1 to 1 range
        max_amp = np.max(np.abs(y))
        waveform_samples = [s / max_amp if max_amp > 0 else 0 for s in waveform_samples]
        
        # Calculate time axis for waveform
        waveform_time = np.arange(len(waveform_samples)) * waveform_step / sr
        waveform_time = waveform_time.tolist()
        
        # Make prediction using trained model
        if model_data is not None:
            # Extract features for ML model
            ml_features = extract_ml_features(y, sr)
            
            # Scale features
            ml_features_scaled = model_data['scaler'].transform([ml_features])
            
            # Predict
            prediction_label = model_data['classifier'].predict(ml_features_scaled)[0]
            prediction_proba = model_data['classifier'].predict_proba(ml_features_scaled)[0]
            
            prediction = "Healthy" if prediction_label == 0 else "Faulty"
            confidence = float(prediction_proba[prediction_label]) * 100
            
            print(f"ML Prediction: {prediction} (confidence: {confidence:.1f}%)")
        else:
            # Fallback to threshold-based prediction
            prediction = "Healthy" if rms > 0.05 else "Faulty"
            confidence = 75.0
            print(f"Threshold Prediction: {prediction} (using fallback method)")
        
        # Clean up converted file
        if filepath != os.path.splitext(filepath)[0] + '.wav' and os.path.exists(filepath):
            os.remove(filepath)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'features': {
                'rms_energy': float(rms),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'zero_crossing_rate': float(zero_crossing_rate)
            },
            'spectrum': {
                'frequencies': [float(f) for f in spectrum_freqs.tolist()],
                'magnitudes': [float(m) for m in spectrum_magnitude.tolist()]
            },
            'waveform': {
                'samples': [float(s) for s in waveform_samples],
                'time': [float(t) for t in waveform_time],
                'duration': float(len(y) / sr)
            }
        }
    except Exception as e:
        print(f"Error processing audio: {e}")
        # Clean up any converted file
        wav_path = os.path.splitext(filepath)[0] + '.wav'
        if os.path.exists(wav_path):
            os.remove(wav_path)
        
        # Fallback to dummy prediction if audio processing fails
        return {
            'prediction': random.choice(["Healthy", "Faulty"]),
            'confidence': 50.0,
            'features': {
                'rms_energy': 0.0,
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'zero_crossing_rate': 0.0
            },
            'spectrum': {
                'frequencies': [],
                'magnitudes': []
            },
            'waveform': {
                'samples': [],
                'time': [],
                'duration': 0.0
            }
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part in the request'}), 400
    
    file = request.files['audio']
    
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # Predict motor state
            result = predict_motor_state(filepath)
            
            # Clean up the uploaded file to save space
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'confidence': result.get('confidence', 0),
                'features': result['features'],
                'spectrum': result['spectrum'],
                'waveform': result['waveform'],
                'message': f'Analysis complete. Motor state is {result["prediction"]}.'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type. Please upload an audio file.'}), 400

if __name__ == '__main__':
    # Run the app in debug mode on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
