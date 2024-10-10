import os
import sys
import random
import librosa
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal

input_folder = sys.argv[1]
output_folder = sys.argv[2]

os.makedirs(output_folder, exist_ok=True)

def add_noise(data, noise_factor=0.005):
    """Adds white noise to audio signal."""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type to avoid distortions
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def time_stretch(data, rate=1.1):
    """Stretches the audio signal."""
    return librosa.effects.time_stretch(data, rate)

def pitch_shift(data, sr, n_steps=2):
    """Shifts the pitch of the audio signal."""
    return librosa.effects.pitch_shift(data, sr, n_steps=n_steps)

def noise_reduction(data, sr):
    """Applies noise reduction using spectral gating."""
    reduced_noise = signal.wiener(data)
    return reduced_noise

def augment_and_save(file_path, output_dir):
    """Applies all augmentations to an audio file and saves the results."""
    filename = os.path.basename(file_path)
    base_filename = os.path.splitext(filename)[0]
    
    audio, sr = librosa.load(file_path, sr=None)

    # Original
    librosa.output.write_wav(os.path.join(output_dir, f"{base_filename}_original.wav"), audio, sr)
    
    # Add noise
    noisy_audio = add_noise(audio)
    librosa.output.write_wav(os.path.join(output_dir, f"{base_filename}_noisy.wav"), noisy_audio, sr)

    # Time stretch
    stretched_audio = time_stretch(audio)
    librosa.output.write_wav(os.path.join(output_dir, f"{base_filename}_stretched.wav"), stretched_audio, sr)

    # Pitch shift
    pitched_audio = pitch_shift(audio, sr)
    librosa.output.write_wav(os.path.join(output_dir, f"{base_filename}_pitched.wav"), pitched_audio, sr)

    # Noise reduction
    reduced_noise_audio = noise_reduction(audio, sr)
    librosa.output.write_wav(os.path.join(output_dir, f"{base_filename}_cleaned.wav"), reduced_noise_audio, sr)

    print(f"Augmentations and noise reduction applied to {filename}.")

# Apply augmentations to each file in the input folder
for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            augment_and_save(file_path, output_folder)


