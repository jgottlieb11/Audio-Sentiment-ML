import collections
import contextlib
import sys
import wave
import librosa
import webrtcvad
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from copy import deepcopy

# Referred to https://github.com/wiseman/py-webrtcvad/blob/master/example.py

# Function to read a wave (.wav) file
def load_wav_file(filepath):
    """Reads a .wav file from the given path.

    Args:
        filepath (str): Path to the .wav file.

    Returns:
        tuple: PCM audio data and sample rate.
    """
    with contextlib.closing(wave.open(filepath, 'rb')) as wav_file:
        num_channels = wav_file.getnchannels()
        assert num_channels == 1, "Input audio must be mono."
        sample_width = wav_file.getsampwidth()
        assert sample_width == 2, "Only 16-bit audio is supported."
        sample_rate = wav_file.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000), "Unsupported sample rate."
        pcm_audio_data = wav_file.readframes(wav_file.getnframes())
        return pcm_audio_data, sample_rate


# Function to write a .wav file
def save_wav_file(output_path, pcm_audio, sample_rate):
    """Saves a .wav file to the given path.

    Args:
        output_path (str): Destination file path.
        pcm_audio (bytes): PCM audio data to write.
        sample_rate (int): Audio sample rate in Hz.
    """
    with contextlib.closing(wave.open(output_path, 'wb')) as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio)


# Frame class to represent audio frames
class AudioFrame:
    """Represents a single audio frame with PCM data."""
    
    def __init__(self, audio_bytes, timestamp, frame_duration):
        self.audio_bytes = audio_bytes
        self.timestamp = timestamp
        self.frame_duration = frame_duration


# Function to generate frames from PCM audio data
def generate_audio_frames(frame_ms, pcm_audio, sample_rate):
    """Generates audio frames of a given duration.

    Args:
        frame_ms (float): Frame duration in milliseconds.
        pcm_audio (bytes): PCM audio data.
        sample_rate (int): Audio sample rate in Hz.

    Yields:
        AudioFrame: Yields instances of AudioFrame.
    """
    frame_size = int(sample_rate * (frame_ms / 1000.0) * 2)
    start = 0
    timestamp = 0.0
    duration = (float(frame_size) / sample_rate) / 2.0
    while start + frame_size < len(pcm_audio):
        yield AudioFrame(pcm_audio[start:start + frame_size], timestamp, duration)
        timestamp += duration
        start += frame_size


# Function to collect voiced frames using VAD (Voice Activity Detection)
def collect_voiced_frames(sample_rate, frame_ms, padding_ms, vad_instance, audio_frames):
    """Collects voiced frames from the audio.

    Args:
        sample_rate (int): Audio sample rate in Hz.
        frame_ms (int): Frame duration in milliseconds.
        padding_ms (int): Padding duration in milliseconds.
        vad_instance (webrtcvad.Vad): Instance of VAD.
        audio_frames (list): List of AudioFrame instances.

    Yields:
        bytes: Yields the voiced audio frames as PCM bytes.
    """
    padding_frame_count = int(padding_ms / frame_ms)
    sliding_window = collections.deque(maxlen=padding_frame_count)
    triggered = False
    voiced_audio = []

    for frame in audio_frames:
        is_voiced = vad_instance.is_speech(frame.audio_bytes, sample_rate)

        sys.stdout.write('1' if is_voiced else '0')
        if not triggered:
            sliding_window.append((frame, is_voiced))
            voiced_frame_count = len([f for f, speech in sliding_window if speech])
            if voiced_frame_count > 0.9 * sliding_window.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (sliding_window[0][0].timestamp,))
                for f, _ in sliding_window:
                    voiced_audio.append(f)
                sliding_window.clear()
        else:
            voiced_audio.append(frame)
            sliding_window.append((frame, is_voiced))
            unvoiced_frame_count = len([f for f, speech in sliding_window if not speech])
            if unvoiced_frame_count > 0.9 * sliding_window.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.frame_duration))
                triggered = False
                yield b''.join([f.audio_bytes for f in voiced_audio])
                sliding_window.clear()
                voiced_audio = []

    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.frame_duration))

    if voiced_audio:
        yield b''.join([f.audio_bytes for f in voiced_audio])


# Function to calculate MFCC and related features for audio data
def extract_mfcc_features(audio_data, sampling_rate, n_mfcc=13, n_fft=0.032, hop_length=0.010):
    """Extract MFCC and related features from the audio.

    Args:
        audio_data (np.array): The audio signal.
        sampling_rate (int): Audio sample rate.
        n_mfcc (int, optional): Number of MFCC features to extract. Defaults to 13.
        n_fft (float, optional): FFT window size. Defaults to 0.032.
        hop_length (float, optional): Hop length in seconds. Defaults to 0.010.

    Returns:
        np.array: Combined MFCC, delta, and delta-delta features.
    """
    mfcc = librosa.feature.mfcc(audio_data, sr=sampling_rate, hop_length=int(hop_length * sampling_rate), n_fft=int(n_fft * sampling_rate), n_mfcc=n_mfcc, dct_type=2)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    combined_features = np.vstack((mfcc, delta_mfcc, delta2_mfcc))
    return combined_features.T


# Function to train a UBM using a Gaussian Mixture Model (GMM)
def train_ubm_model(features, n_components=16, covariance_type='full'):
    """Trains a UBM using a GMM model on the provided features.

    Args:
        features (np.array): Input features for GMM training.
        n_components (int, optional): Number of GMM components. Defaults to 16.
        covariance_type (str, optional): Type of covariance. Defaults to 'full'.

    Returns:
        GaussianMixture: The trained GMM model.
    """
    ubm_model = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    ubm_model.fit(features)
    return ubm_model


# Function to perform MAP adaptation for a GMM
def perform_map_adaptation(gmm_model, audio_features, iterations=1, relevance_factor=16):
    """Performs Maximum A Posteriori (MAP) adaptation on a GMM model.

    Args:
        gmm_model (GaussianMixture): The UBM model.
        audio_features (np.array): Features from the audio data.
        iterations (int, optional): Number of adaptation iterations. Defaults to 1.
        relevance_factor (int, optional): Relevance factor for adaptation. Defaults to 16.

    Returns:
        GaussianMixture: The adapted GMM model.
    """
    num_data = audio_features.shape[0]
    num_features = audio_features.shape[1]
    num_components = gmm_model.n_components

    updated_means = np.zeros((num_components, num_features))
    n_k = np.zeros((num_components, 1))

    mu_k = gmm_model.means_
    pi_k = gmm_model.weights_

    log_likelihood_old = gmm_model.score(audio_features)
    log_likelihood_new = 0
    iteration_count = 0

    while iteration_count < iterations:
        iteration_count += 1
        log_likelihood_old = log_likelihood_new
        z_n_k = gmm_model.predict_proba(audio_features)
        n_k = np.sum(z_n_k, axis=0).reshape(np.shape(n_k)[0], 1)
        updated_means = np.dot(z_n_k.T, audio_features) / n_k
        adaptation_coeff = n_k / (n_k + relevance_factor)
        I = np.ones_like(n_k)
        mu_k = (adaptation_coeff * updated_means) + ((I - adaptation_coeff) * mu_k)
        gmm_model.means_ = mu_k
        log_likelihood_new = gmm_model.score(audio_features)

        if abs(log_likelihood_old - log_likelihood_new) < 1e-20:
            break

    return gmm_model


########################### IMPLEMENTATION ###########################
# Read audio data and process chunks
pcm_audio, sample_rate = load_wav_file('test.wav')
vad = webrtcvad.Vad(2)  # Level 2 VAD
audio_frames = list(generate_audio_frames(30, pcm_audio, sample_rate))
voiced_segments = collect_voiced_frames(sample_rate, 30, 300, vad, audio_frames)

# Save voiced audio chunks
chunk_count = 0
for idx, segment in enumerate(voiced_segments):
    chunk_filename = f'chunk-{idx:02d}.wav'
    save_wav_file(chunk_filename, segment, sample_rate)
    chunk_count += 1

# GMM feature extraction and clustering
ubm_audio, ubm_sr = librosa.load(sys.argv[1])
ubm_features = extract_mfcc_features(ubm_audio, ubm_sr)

# Train UBM with Gaussian Mixture Model
ubm_model = train_ubm_model(ubm_features)
print("UBM model score:", ubm_model.score(ubm_features))

# MAP adaptation for individual chunks
total_sv = []
for i in range(chunk_count):
    chunk_filename = f'chunk-{i:02d}.wav'
    chunk_audio, chunk_sr = librosa.load(chunk_filename, sr=None)
    chunk_features = extract_mfcc_features(chunk_audio, chunk_sr)
    
    gmm_adapted = deepcopy(ubm_model)
    gmm_adapted = perform_map_adaptation(gmm_adapted, chunk_features, iterations=1)
    
    sv = gmm_adapted.means_.flatten()
    total_sv.append(sv)

# Clustering using spectral clustering
n_clusters = 2
cluster_labels = SpectralClustering(n_clusters=n_clusters, affinity='cosine').fit_predict(total_sv)

# Re-arrange labels for consistency
def rearrange_labels(labels, n):
    unique_labels = sorted(set(labels))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    return [label_mapping[label] for label in labels]

cluster_labels = rearrange_labels(cluster_labels, n_clusters)
print("Cluster labels:", cluster_labels)

# Identify speaker turns (e.g., customer and agent in a call)
print("Speaker turns for customer:", [i for i, label in enumerate(cluster_labels) if label == 1])
