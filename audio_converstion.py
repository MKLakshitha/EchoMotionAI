import sounddevice as sd
from scipy.io.wavfile import write

# Parameters for the recording
fs = 44100  # Sample rate (standard is 44100 Hz)
duration = 3  # Duration in seconds
filename = "recorded_audio.wav"  # Name of the output file

print("Recording...")

# Record audio using the microphone
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()  # Wait until recording is complete

# Save the recorded audio to a file
write(filename, fs, audio_data)

print(f"Recording complete. Audio saved as '{filename}'")
