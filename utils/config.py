import os

# Audio Recording Parameters
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1  # Mono
CHUNK_SIZE = 1024
DURATION = 10  # seconds

# VAD Parameters
VAD_FRAME_DURATION = 20  # ms
VAD_MODE = 3  # Aggressiveness mode (0-3)

# Diarization Parameters
DIARIZATION_MIN_SPEAKERS = 1
DIARIZATION_MAX_SPEAKERS = 5

# File Paths
AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audio")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

# Create directories if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
