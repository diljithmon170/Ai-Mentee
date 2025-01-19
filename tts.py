import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np  # Import numpy

# Load the audio file
audio_path = 'audio.wav.opus'  # Replace with your file path
y, sr = librosa.load(audio_path, sr=None)

# Generate the spectrogram
S = librosa.stft(y)  # Short-time Fourier transform
S_db = librosa.amplitude_to_db(abs(S), ref=np.max)  # Convert to decibels

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()
