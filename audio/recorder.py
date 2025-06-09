import sounddevice as sd
import numpy as np
from utils.config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE, DURATION
import queue
import threading
import time

class AudioRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print(f"Status: {status}")
        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()
        self.audio_queue.put(audio_data.copy())

    def start_recording(self):
        """Start recording audio in a separate thread"""
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        """Stop recording and return the recorded audio"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        # Combine all chunks from the queue
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())
        
        if audio_chunks:
            return np.concatenate(audio_chunks)
        return None

    def _record_audio(self):
        """Internal method to handle the recording process"""
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE,
                              channels=CHANNELS,
                              callback=self.audio_callback,
                              blocksize=CHUNK_SIZE,
                              dtype=np.float32):
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error during recording: {str(e)}")
            self.is_recording = False

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Record audio for a specified duration
    
    Args:
        duration (int): Recording duration in seconds
        sample_rate (int): Audio sample rate in Hz
    
    Returns:
        numpy.ndarray: Recorded audio data
    """
    recorder = AudioRecorder()
    print(f"Recording for {duration} seconds...")
    
    # Start recording
    recorder.start_recording()
    time.sleep(duration)
    
    # Stop recording and get the audio data
    audio_data = recorder.stop_recording()
    
    if audio_data is None:
        raise RuntimeError("No audio data was recorded")
    
    # Normalize audio to prevent clipping
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    print("Recording completed")
    return audio_data