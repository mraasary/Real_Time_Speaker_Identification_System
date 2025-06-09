import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import time

class AudioIO:
    def __init__(self):
        self.sample_rate = 16000
    
    def get_intel_microphone(self):
        """
        Find and return the Intel Smart Sound Technology microphone device
        """
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if "Intel" in device['name'] and device['max_input_channels'] > 0:
                return i
        return None
    
    def play_test_tone(self, duration=2, frequency=440):
        """
        Play a test tone to verify playback device
        """
        print("Playing test tone...")
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        tone = 0.2 * np.sin(2 * np.pi * frequency * t)
        sd.play(tone, self.sample_rate)
        sd.wait()
        print("Test tone playback complete!")
    
    def list_audio_devices(self):
        """
        List all available audio input devices
        """
        devices = sd.query_devices()
        print("\nAvailable Audio Input Devices:")
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']} (Input channels: {device['max_input_channels']}, Default rate: {device['default_samplerate']})")
                input_devices.append(i)
        return input_devices
    
    def to_mono(self, audio):
        """
        Convert multi-channel audio to mono by averaging channels
        """
        if audio.ndim == 1:
            return audio
        return np.mean(audio, axis=1)
    
    def select_input_device(self):
        input_devices = self.list_audio_devices()
        if not input_devices:
            print("No input devices found!")
            return None
        try:
            idx = int(input(f"Select input device index from the list above [{input_devices[0]}]: ") or input_devices[0])
            if idx not in input_devices:
                print("Invalid selection. Using default input device.")
                return None
            return idx
        except Exception as e:
            print(f"Error selecting device: {e}. Using default input device.")
            return None
    
    def record_audio(self, device_index=None, duration=5):
        """
        Record audio from microphone
        """
        try:
            if device_index is not None:
                print(f"\nUsing microphone: {sd.query_devices(device_index)['name']}")
                sd.default.device = (device_index, None)
            print(f"\nRecording for {duration} seconds...")
            print("Please speak into your microphone...")
            audio = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float32',
                          blocking=True)
            if audio is None or len(audio) == 0:
                raise Exception("No audio data was recorded")
            print(f"Recording complete! Recorded {len(audio)/self.sample_rate:.2f} seconds of audio")
            max_amplitude = np.max(np.abs(audio))
            print(f"Maximum audio amplitude: {max_amplitude:.4f}")
            if max_amplitude < 0.01:
                print("Warning: The recorded audio seems to be very quiet or silent")
            return self.to_mono(audio)
        except Exception as e:
            print(f"Error during recording: {str(e)}")
            return None
    
    def save_audio(self, audio, filename):
        """
        Save audio to a WAV file
        """
        try:
            if audio is None or len(audio) == 0:
                raise Exception("No audio data to save")
            audio_mono = self.to_mono(audio)
            sf.write(filename, audio_mono, self.sample_rate)
            print(f"Audio saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            return False
    
    def load_audio(self, filename):
        """
        Load audio from a WAV file
        """
        try:
            audio, sample_rate = sf.read(filename)
            return audio, sample_rate
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            return None, None
    
    def play_audio_file(self, filename):
        """
        Play audio from a WAV file
        """
        try:
            audio, sample_rate = self.load_audio(filename)
            if audio is None:
                return
            audio_mono = self.to_mono(audio)
            print(f"Playing {filename}...")
            sd.play(audio_mono, sample_rate)
            sd.wait()
            print("Playback complete!")
        except Exception as e:
            print(f"Error playing audio: {str(e)}") 