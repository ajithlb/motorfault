"""
Generate synthetic motor sound samples for training.
Creates 10 healthy and 10 faulty motor sounds with realistic variations.
"""

import numpy as np
import librosa
import soundfile as sf
import os
from scipy import signal

def generate_healthy_motor_sound(duration=5, sr=22050, variation_seed=None):
    """
    Generate a healthy motor sound with low-frequency hum and minimal noise.
    
    Characteristics of healthy motors:
    - Steady low-frequency fundamental (50-200 Hz)
    - Minimal harmonics
    - Low noise floor
    - Consistent amplitude
    """
    if variation_seed is not None:
        np.random.seed(variation_seed)
    
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base frequency with slight variation (50-150 Hz for healthy motor)
    base_freq = np.random.uniform(60, 120)
    freq_variation = np.random.uniform(0.5, 2.0)  # Slight frequency drift
    
    # Generate fundamental frequency with slight modulation
    fundamental = 0.3 * np.sin(2 * np.pi * base_freq * t + 
                                0.1 * np.sin(2 * np.pi * freq_variation * t))
    
    # Add 2nd harmonic (weaker)
    harmonic_2 = 0.1 * np.sin(2 * np.pi * (2 * base_freq) * t)
    
    # Add 3rd harmonic (even weaker)
    harmonic_3 = 0.05 * np.sin(2 * np.pi * (3 * base_freq) * t)
    
    # Add very subtle mechanical noise
    noise = np.random.normal(0, 0.02, len(t))
    
    # Low-pass filter the noise to make it more realistic
    b, a = signal.butter(4, 500 / (sr / 2), btype='low')
    noise = signal.filtfilt(b, a, noise)
    
    # Combine all components
    audio = fundamental + harmonic_2 + harmonic_3 + noise
    
    # Add slight amplitude modulation (breathing effect)
    modulation = 1 + 0.05 * np.sin(2 * np.pi * 0.5 * t)
    audio = audio * modulation
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.7
    
    return audio

def generate_faulty_motor_sound(duration=5, sr=22050, variation_seed=None):
    """
    Generate a faulty motor sound with irregular patterns and high-frequency noise.
    
    Characteristics of faulty motors:
    - Higher fundamental frequency
    - Strong harmonics and sub-harmonics
    - Irregular amplitude (bearing wear, imbalance)
    - High-frequency noise (friction, loose parts)
    - Possible knocking/clicking sounds
    """
    if variation_seed is not None:
        np.random.seed(variation_seed)
    
    t = np.linspace(0, duration, int(sr * duration))
    
    # Higher base frequency with more variation (faulty motors run hotter/faster)
    base_freq = np.random.uniform(100, 200)
    freq_variation = np.random.uniform(2.0, 5.0)  # More frequency instability
    
    # Generate fundamental with irregular modulation
    fundamental = 0.35 * np.sin(2 * np.pi * base_freq * t + 
                                 0.3 * np.sin(2 * np.pi * freq_variation * t))
    
    # Add stronger harmonics (indicating mechanical issues)
    harmonic_2 = 0.2 * np.sin(2 * np.pi * (2 * base_freq) * t)
    harmonic_3 = 0.15 * np.sin(2 * np.pi * (3 * base_freq) * t)
    harmonic_4 = 0.1 * np.sin(2 * np.pi * (4 * base_freq) * t)
    
    # Add sub-harmonic (bearing defect signature)
    subharmonic = 0.12 * np.sin(2 * np.pi * (base_freq / 2) * t)
    
    # Add high-frequency noise (friction, loose components)
    noise = np.random.normal(0, 0.08, len(t))
    
    # High-pass filter some noise for metallic sound
    b_high, a_high = signal.butter(4, 1000 / (sr / 2), btype='high')
    high_freq_noise = signal.filtfilt(b_high, a_high, np.random.normal(0, 0.06, len(t)))
    
    # Add random "knocking" sounds (bearing defects)
    num_knocks = np.random.randint(5, 15)
    knock_positions = np.random.choice(len(t), num_knocks, replace=False)
    knocks = np.zeros(len(t))
    for pos in knock_positions:
        if pos < len(t) - 1000:
            # Create a sharp transient
            knock_envelope = np.exp(-np.arange(1000) / 100)
            knocks[pos:pos+1000] += knock_envelope * np.random.uniform(0.3, 0.5)
    
    # Combine all components
    audio = (fundamental + harmonic_2 + harmonic_3 + harmonic_4 + 
             subharmonic + noise + high_freq_noise + knocks)
    
    # Add irregular amplitude modulation (imbalance, wobble)
    modulation = 1 + 0.15 * np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.sin(2 * np.pi * 3.7 * t)
    audio = audio * modulation
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio

def create_training_dataset():
    """
    Create a complete training dataset with 10 healthy and 10 faulty motor sounds.
    """
    print("=" * 70)
    print("Generating Synthetic Motor Sound Dataset")
    print("=" * 70)
    
    # Create directories
    healthy_dir = "training_data/healthy"
    faulty_dir = "training_data/faulty"
    
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(faulty_dir, exist_ok=True)
    
    sr = 22050  # Sample rate
    duration = 5  # 5 seconds per sample
    
    print(f"\nGenerating audio files:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {duration} seconds")
    print(f"  Format: WAV (16-bit)")
    
    # Generate healthy motor sounds
    print(f"\n📁 Creating healthy motor sounds in '{healthy_dir}/'...")
    for i in range(1, 11):
        filename = f"{healthy_dir}/healthy_motor_{i:02d}.wav"
        audio = generate_healthy_motor_sound(duration=duration, sr=sr, variation_seed=i)
        sf.write(filename, audio, sr, subtype='PCM_16')
        print(f"  ✓ Generated: healthy_motor_{i:02d}.wav")
    
    # Generate faulty motor sounds
    print(f"\n📁 Creating faulty motor sounds in '{faulty_dir}/'...")
    for i in range(1, 11):
        filename = f"{faulty_dir}/faulty_motor_{i:02d}.wav"
        audio = generate_faulty_motor_sound(duration=duration, sr=sr, variation_seed=i + 100)
        sf.write(filename, audio, sr, subtype='PCM_16')
        print(f"  ✓ Generated: faulty_motor_{i:02d}.wav")
    
    print("\n" + "=" * 70)
    print("Dataset Generation Complete!")
    print("=" * 70)
    
    print(f"\n📊 Summary:")
    print(f"  Healthy samples: 10 files in '{healthy_dir}/'")
    print(f"  Faulty samples:  10 files in '{faulty_dir}/'")
    print(f"  Total samples:   20 files")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Review the generated audio files")
    print(f"  2. Run: python train_model_extended.py")
    print(f"  3. Test the improved model")
    
    print("\n💡 Tip: You can listen to the generated files to verify they sound realistic!")
    print("=" * 70)

if __name__ == '__main__':
    try:
        create_training_dataset()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
