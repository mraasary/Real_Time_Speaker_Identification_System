import webrtcvad
import numpy as np
import soundfile as sf
from typing import List, Tuple

class VADProcessor:
    def __init__(self, aggressiveness=3):
        """
        Initialize VAD processor
        aggressiveness: 0-3, higher means more aggressive in filtering out non-speech
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration = 20  # ms
        self.sample_rate = 16000  # Hz

    def is_speech(self, frame: bytes) -> bool:
        """
        Check if a single frame contains speech
        """
        return self.vad.is_speech(frame, self.sample_rate)

    def process_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
        """
        Process audio and return speech segments and speech flags
        """
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Calculate frame size
        frame_size = int(self.sample_rate * self.frame_duration / 1000)
        num_frames = len(audio_int16) // frame_size
        
        # Process each frame
        speech_flags = []
        for i in range(num_frames):
            frame = audio_int16[i * frame_size:(i + 1) * frame_size]
            frame_bytes = frame.tobytes()
            is_speech = self.is_speech(frame_bytes)
            speech_flags.append(is_speech)
        
        return audio_int16, speech_flags

    def extract_speech_segments(self, audio: np.ndarray, speech_flags: List[bool]) -> np.ndarray:
        """
        Extract only the speech segments from the audio
        """
        frame_size = int(self.sample_rate * self.frame_duration / 1000)
        speech_segments = []
        
        for i, is_speech in enumerate(speech_flags):
            if is_speech:
                start = i * frame_size
                end = (i + 1) * frame_size
                speech_segments.append(audio[start:end])
        
        if speech_segments:
            return np.concatenate(speech_segments)
        return np.array([])

    def save_speech_segments(self, audio: np.ndarray, output_file: str):
        """
        Process audio and save only speech segments
        """
        # Process audio
        audio_int16, speech_flags = self.process_audio(audio)
        
        # Extract speech segments
        speech_audio = self.extract_speech_segments(audio_int16, speech_flags)
        
        if len(speech_audio) > 0:
            # Convert back to float32 for saving
            speech_audio_float = speech_audio.astype(np.float32) / 32767.0
            sf.write(output_file, speech_audio_float, self.sample_rate)
            print(f"Speech segments saved to {output_file}")
            return True
        else:
            print("No speech segments detected")
            return False 