import webrtcvad
import numpy as np
from utils.config import SAMPLE_RATE, VAD_MODE

class VAD:
    def __init__(self, mode=2):  # Using mode 2 for balanced sensitivity
        """
        Initialize Voice Activity Detection
        
        Args:
            mode (int): VAD aggressiveness mode (0-3)
                      0: Least aggressive
                      3: Most aggressive
        """
        self.vad = webrtcvad.Vad(mode)
        self.frame_duration = 20  # ms, using 20ms frames for better detection
        self.frame_size = int(SAMPLE_RATE * self.frame_duration / 1000)
        
    def is_speech(self, audio_bytes, sample_rate=SAMPLE_RATE):
        """
        Check if the given audio frame contains speech
        
        Args:
            audio_bytes (bytes): Audio data in bytes
            sample_rate (int): Sample rate of the audio
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        try:
            # Ensure audio is the correct length
            if len(audio_bytes) != self.frame_size * 2:  # 2 bytes per sample
                # Pad or truncate to correct length
                if len(audio_bytes) < self.frame_size * 2:
                    audio_bytes = audio_bytes + b'\x00' * (self.frame_size * 2 - len(audio_bytes))
                else:
                    audio_bytes = audio_bytes[:self.frame_size * 2]
            
            return self.vad.is_speech(audio_bytes, sample_rate)
        except Exception as e:
            print(f"VAD Error: {str(e)}")
            return False
    
    def process_audio(self, audio_data, sample_rate=SAMPLE_RATE):
        """
        Process audio data and return speech segments
        
        Args:
            audio_data (numpy.ndarray): Audio data
            sample_rate (int): Sample rate of the audio
            
        Returns:
            list: List of speech segments
        """
        # Convert to 16-bit PCM
        audio_data = np.clip(audio_data, -1.0, 1.0)  # Ensure values are in [-1, 1]
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Process in frames
        speech_segments = []
        current_segment = []
        in_speech = False
        silence_frames = 0
        max_silence_frames = 10  # Allow more silence between speech segments
        min_segment_frames = 5  # Minimum frames for a valid speech segment
        
        for i in range(0, len(audio_data), self.frame_size):
            frame = audio_data[i:i + self.frame_size]
            if len(frame) != self.frame_size:
                continue
                
            frame_bytes = frame.tobytes()
            is_speech = self.is_speech(frame_bytes, sample_rate)
            
            if is_speech:
                if silence_frames > 0:
                    # Add silence frames to maintain continuity
                    current_segment.extend([0] * silence_frames * self.frame_size)
                current_segment.extend(frame)
                in_speech = True
                silence_frames = 0
            elif in_speech:
                silence_frames += 1
                if silence_frames > max_silence_frames:
                    if current_segment and len(current_segment) >= min_segment_frames * self.frame_size:
                        speech_segments.append(np.array(current_segment))
                    current_segment = []
                    in_speech = False
                    silence_frames = 0
                else:
                    current_segment.extend(frame)
        
        # Add final segment if in speech
        if in_speech and current_segment and len(current_segment) >= min_segment_frames * self.frame_size:
            speech_segments.append(np.array(current_segment))
        
        # Convert segments back to float32 and normalize
        processed_segments = []
        for segment in speech_segments:
            # Convert to float32
            segment = segment.astype(np.float32) / 32767.0
            # Normalize segment
            max_val = np.max(np.abs(segment))
            if max_val > 0:
                segment = segment / max_val
            processed_segments.append(segment)
        
        return processed_segments