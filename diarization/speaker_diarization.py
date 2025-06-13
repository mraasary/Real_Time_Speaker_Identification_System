import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import numpy as np
import soundfile as sf
from utils.config import SAMPLE_RATE, DIARIZATION_MIN_SPEAKERS, DIARIZATION_MAX_SPEAKERS
import os
import requests
import webbrowser

class SpeakerDiarizer:
    def __init__(self):
        # Get Hugging Face token from environment variable
        hf_token = os.getenv('HF_TOKEN', 'Your hugging face token')
        
        try:
            # Check if user has accepted the agreement
            headers = {"Authorization": f"Bearer {hf_token}"}
            response = requests.get(
                "https://huggingface.co/api/models/pyannote/speaker-diarization-3.1",
                headers=headers
            )
            
            if response.status_code == 401:
                print("\nAuthentication failed. Please check your token.")
                print("Visit https://hf.co/settings/tokens to create a new token.")
                raise Exception("Invalid Hugging Face token")
            
            if response.status_code == 403:
                print("\nYou need to accept the user agreement first!")
                print("Opening browser to accept the agreement...")
                webbrowser.open("https://hf.co/pyannote/speaker-diarization-3.1")
                print("\nPlease:")
                print("1. Click 'Access repository'")
                print("2. Accept the terms and conditions")
                print("3. Run the program again")
                raise Exception("User agreement not accepted")
            
            # Initialize the pipeline with authentication
            print("Initializing speaker diarization model...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            print("Speaker diarization model initialized successfully!")
            
        except Exception as e:
            print(f"\nError initializing speaker diarization: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Make sure you've accepted the user agreement at https://hf.co/pyannote/speaker-diarization-3.1")
            print("2. Verify your token is correct")
            print("3. Try setting the token in your environment:")
            print("   - Windows: set HF_TOKEN=your_token_here")
            print("   - Linux/Mac: export HF_TOKEN=your_token_here")
            raise

    def process_audio(self, audio_data, sample_rate=SAMPLE_RATE, num_speakers=None):
        """
        Process audio data to detect and separate speakers
        
        Args:
            audio_data (numpy.ndarray): Audio data
            sample_rate (int): Sample rate of the audio
            num_speakers (int, optional): Number of speakers to detect
        Returns:
            dict: Dictionary containing speaker segments and their timestamps
        """
        # Save audio to temporary file
        temp_file = "temp_audio.wav"
        sf.write(temp_file, audio_data, sample_rate)
        
        try:
            # Process the audio file
            if num_speakers is not None:
                diarization = self.pipeline(temp_file, num_speakers=num_speakers)
            else:
                diarization = self.pipeline(temp_file)
            
            # Extract speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'speaker': speaker,
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.duration
                })
            
            return {
                'num_speakers': len(set(seg['speaker'] for seg in speaker_segments)),
                'segments': speaker_segments
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

def get_speaker_segments(audio_data, sample_rate, num_speakers=None):
    """
    Get speaker segments from audio data
    
    Args:
        audio_data (numpy.ndarray): Audio data
        sample_rate (int): Sample rate of the audio
        num_speakers (int, optional): Number of speakers to detect
    Returns:
        dict: Dictionary containing speaker segments and their timestamps
    """
    diarizer = SpeakerDiarizer()
    return diarizer.process_audio(audio_data, sample_rate, num_speakers=num_speakers) 
