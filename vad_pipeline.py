from audio.recorder import record_audio
from audio.noise_reduction import reduce_noise
from vad.vad import VAD
from diarization.speaker_diarization import get_speaker_segments
from utils.config import SAMPLE_RATE, DURATION, VAD_FRAME_DURATION, OUTPUT_DIR
import soundfile as sf
import numpy as np
import os
from datetime import datetime
from audio.audio_io import AudioIO

def verify_audio(audio, sample_rate):
    """Verify audio data is valid"""
    if audio is None or len(audio) == 0:
        raise ValueError("No audio data recorded")
    if not isinstance(audio, np.ndarray):
        raise ValueError("Audio data must be numpy array")
    if np.isnan(audio).any() or np.isinf(audio).any():
        raise ValueError("Audio data contains invalid values")
    print(f"Audio verification passed: {len(audio)/sample_rate:.2f} seconds of audio")

def run_vad_pipeline():
    """
    Run the complete pipeline:
    1. Record audio
    2. Reduce noise
    3. Detect voice activity
    4. Perform speaker diarization
    """
    try:
        # Initialize AudioIO
        audio_io = AudioIO()
        
        # Step 1: Record Audio
        print("\n=== Step 1: Recording Audio ===")
        print(f"Recording for {DURATION} seconds at {SAMPLE_RATE}Hz...")
        audio = record_audio(duration=DURATION, sample_rate=SAMPLE_RATE)
        verify_audio(audio, SAMPLE_RATE)
        
        # Save original recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_file = os.path.join(OUTPUT_DIR, f"original_{timestamp}.wav")
        audio_io.save_audio(audio, original_file)
        print("\nPlaying original recording...")
        audio_io.play_audio_file(original_file)

        # Step 2: Reduce Noise
        print("\n=== Step 2: Noise Reduction ===")
        print("Applying noise reduction...")
        clean_audio = reduce_noise(audio, sample_rate=SAMPLE_RATE)
        verify_audio(clean_audio, SAMPLE_RATE)
        print("Noise reduction completed")
        
        # Save cleaned audio
        cleaned_file = os.path.join(OUTPUT_DIR, f"cleaned_{timestamp}.wav")
        audio_io.save_audio(clean_audio, cleaned_file)
        print("\nPlaying cleaned audio...")
        audio_io.play_audio_file(cleaned_file)

        # Step 3: Voice Activity Detection
        print("\n=== Step 3: Voice Activity Detection ===")
        vad = VAD()
        speech_segments = vad.process_audio(clean_audio, SAMPLE_RATE)
        
        if not speech_segments:
            print("No speech detected")
            return

        print(f"Detected {len(speech_segments)} speech segments")
        total_speech_duration = sum(len(seg) for seg in speech_segments) / SAMPLE_RATE
        print(f"Total speech duration: {total_speech_duration:.2f} seconds")
        
        # Combine speech segments
        speech_audio = np.concatenate(speech_segments)
        
        # Save speech audio
        speech_file = os.path.join(OUTPUT_DIR, f"speech_{timestamp}.wav")
        audio_io.save_audio(speech_audio, speech_file)
        print("\nPlaying speech-only segments...")
        audio_io.play_audio_file(speech_file)

        # Step 4: Speaker Diarization
        print("\n=== Step 4: Speaker Diarization ===")
        print("Analyzing speaker segments...")
        diarization_results = get_speaker_segments(speech_audio, SAMPLE_RATE)
        
        # Print results
        print(f"\nResults:")
        print(f"Number of speakers detected: {diarization_results['num_speakers']}")
        print("\nSpeaker segments:")
        for segment in diarization_results['segments']:
            print(f"Speaker {segment['speaker']}: {segment['start']:.2f}s - {segment['end']:.2f}s "
                  f"(duration: {segment['duration']:.2f}s)")

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"\nError in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    run_vad_pipeline()
