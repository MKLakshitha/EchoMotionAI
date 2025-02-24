import os
import time
import keyboard
import wave
import pyaudio
from pathlib import Path

def record_audio(filename="audio_input/input.wav"):
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    print("Press and hold SPACE to record audio")
    print("Release SPACE to stop recording")
    print("Press 'q' to quit the program")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    while True:
        if keyboard.is_pressed('q'):
            print("\nExiting...")
            break
            
        if keyboard.is_pressed('space'):
            print("\nRecording... (Release SPACE to stop)")
            
            # Open stream
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
            
            frames = []
            
            # Record until space is released
            while keyboard.is_pressed('space'):
                data = stream.read(CHUNK)
                frames.append(data)
                time.sleep(0.001)  # Small delay to prevent CPU overload
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            
            print("Recording stopped")
            
            # Save the audio file
            try:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                # Save as WAV file
                with wave.open(filename, 'wb') as wav_file:
                    wav_file.setnchannels(CHANNELS)
                    wav_file.setsampwidth(p.get_sample_size(FORMAT))
                    wav_file.setframerate(RATE)
                    wav_file.writeframes(b''.join(frames))
                
                print(f"Saved audio to {filename}")
                
                # Wait for the file to be accessed
                last_modified = os.path.getmtime(filename)
                while os.path.exists(filename):
                    current_modified = os.path.getmtime(filename)
                    if current_modified != last_modified:
                        print("File was accessed, waiting for processing...")
                        time.sleep(1)  # Give time for the WSL program to read the file
                        try:
                            os.remove(filename)
                            print("File processed and deleted")
                        except:
                            print("File is being processed...")
                        break
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error saving audio: {e}")
        
        time.sleep(0.1)
    
    # Cleanup
    p.terminate()

if __name__ == "__main__":
    record_audio()
