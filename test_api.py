"""
Test the Flask API with sample motor sound files to verify predictions.
"""

import requests
import os

def test_prediction(filepath, expected):
    """Test prediction for a single file."""
    url = 'http://127.0.0.1:5000/predict'
    
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return False
    
    try:
        with open(filepath, 'rb') as f:
            files = {'audio': (os.path.basename(filepath), f, 'audio/wav')}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            
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
        else:
            print(f"✗ HTTP Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server. Is Flask running on http://127.0.0.1:5000?")
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
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the model.")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
