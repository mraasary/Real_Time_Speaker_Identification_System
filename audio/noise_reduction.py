import numpy as np
from scipy import signal
from utils.config import SAMPLE_RATE

def fade_in_out(chunk, fade_len=100):
    if len(chunk) < 2 * fade_len:
        return chunk
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    chunk[:fade_len] *= fade_in
    chunk[-fade_len:] *= fade_out
    return chunk

def reduce_noise(audio, sample_rate=SAMPLE_RATE):
    """
    Reduce noise in audio data using a simple bandpass filter and normalization
    with minimal echo artifacts.
    
    Args:
        audio (numpy.ndarray): Audio data
        sample_rate (int): Sample rate of the audio
        
    Returns:
        numpy.ndarray: Cleaned audio data
    """
    # Convert to float32 to reduce memory usage
    audio = audio.astype(np.float32)
    
    # Normalize audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Design bandpass filter (focus on human speech frequencies)
    nyquist = sample_rate / 2
    low = 100 / nyquist  # 100 Hz (lower cutoff for better bass)
    high = 8000 / nyquist  # 8000 Hz (higher cutoff for better clarity)
    
    # Ensure frequencies are within valid range
    low = max(0.001, min(0.499, low))
    high = max(0.001, min(0.499, high))
    
    b, a = signal.butter(2, [low, high], btype='band')  # Lower order filter
    
    # Calculate minimum chunk size needed for the filter
    min_chunk_size = 3 * max(len(b), len(a))  # Rule of thumb for filter stability
    
    # Process in chunks to reduce memory usage
    chunk_size = max(8000, min_chunk_size)  # Ensure chunk size is large enough
    num_chunks = len(audio) // chunk_size + (1 if len(audio) % chunk_size else 0)
    cleaned_chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(audio))
        chunk = audio[start_idx:end_idx]
        
        # Skip processing if chunk is too small
        if len(chunk) < min_chunk_size:
            cleaned_chunks.append(chunk)
            continue
        
        try:
            # Apply bandpass filter
            cleaned_chunk = signal.filtfilt(b, a, chunk)
            
            # Remove or soften noise gate
            # threshold = 0.002
            # cleaned_chunk[np.abs(cleaned_chunk) < threshold] = 0
            cleaned_chunk = fade_in_out(cleaned_chunk, fade_len=200)
            
            cleaned_chunks.append(cleaned_chunk)
        except ValueError:
            # If filtering fails, use the original chunk
            cleaned_chunks.append(chunk)
    
    # Combine chunks
    cleaned_audio = np.concatenate(cleaned_chunks)
    
    # Restore original scale
    if max_val > 0:
        cleaned_audio = cleaned_audio * max_val
    
    return cleaned_audio
