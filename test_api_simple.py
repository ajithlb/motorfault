"""
Simple test of the Flask API using urllib (built-in).
"""

import urllib.request
import urllib.parse
import json
import os

def test_prediction(filepath, expected):
    """Test prediction for a single file."""
    url = 'http://127.0.0.1:5000/predict'
    
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return False
    
    try:
        # Read file
        with open(filepath, 'rb') as f:
            audio_data = f.read()
        
        # Create multipart form data
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        
        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="audio"; filename="{os.path.basename(filepath)}"\r\n'
            f'Content-Type: audio/wav\r\n\r\n'
        ).encode('utf-8')
        
        body += audio_data
        body += f'\r\n--{boundary}--\r\n'.encode('utf-8')
        
        # Create request
        req = urllib.request.Request(url, data=body)
        req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
        
        # Send request
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        if data.get('success'):
            prediction = data.get('prediction')
            confidence = data.get('confidence', 0)
            features = data.get('features', {})
            
            status = "✓" if prediction == expected else "✗"
            
            print(f"\n{status} Test: {filepath}")
            print(f"  Expected:   {expected}")
            print(f"  Predicted:  {prediction}")
            print(f"  Confidence: {confidence:.1f}%")
            print(f"  Features:")
            print(f"    - RMS Energy: {features.get('rms_energy', 0):.4f}")
            print(f"    - Spectral Centroid: {features.get('spectral_centroid', 0):.2f}")
            print(f"    - Spectral Rolloff: {features.get('spectral_rolloff', 0):.2f}")
            print(f"    - Zero Crossing Rate: {features.get('zero_crossing_rate', 0):.4f}")
            
            return prediction == expected
        else:
            print(f"✗ API Error: {data.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing Motor Fault Detection API")
    print("=" * 70)
    
    tests = [
        ('sample_motor_healthy.wav', 'Healthy'),
        ('sample_motor_faulty.wav', 'Faulty')
    ]
    
    results = []
    for filepath, expected in tests:
        result = test_prediction(filepath, expected)
        results.append(result)
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The model is working correctly!")
        print("\n✅ The Flask application is running successfully!")
        print("✅ ML model predictions are accurate!")
        print("\n📍 Access the web interface at: http://127.0.0.1:5000")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the model.")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
