# 🎙️ Speaker Detection and Identification System

This project detects whether someone is speaking in real-time, extracts speech segments using Voice Activity Detection (VAD), and performs speaker diarization to identify different speakers.

## ✅ Current Features 

- 🔴 Live audio recording via microphone 
- 🧼 Noise reduction using `noisereduce`  
- 🗣️ Voice Activity Detection (VAD) using `webrtcvad`  
- ✂️ Extraction of speech-only segments
- 👥 Speaker diarization using `pyannote-audio`
- 📂 Modular Python codebase  

## 📁 Project Structure

```
speaker_identification/
├── audio/                 # Handles recording and noise reduction
├── vad/                   # Voice Activity Detection module
├── diarization/          # Speaker diarization module
├── utils/                # Config settings
├── output/               # Output directory for processed audio
├── vad_pipeline.py       # Runs recording, noise reduction, VAD, and diarization
├── main.py               # Entry point that calls the pipeline
├── requirements.txt      # Python dependencies
```

## 🚀 How to Run

1. **Clone the repo:**
   ```bash
   git clone https://github.com/mraasary/Real_Time_Speaker_Identification_System.git
   cd Real_Time_Speaker_Identification_System
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Hugging Face token:**
   - Get your token from https://huggingface.co/settings/tokens
   - Set it as an environment variable:
     ```bash
     # Windows
     set HF_TOKEN=your_token_here
     # Linux/Mac
     export HF_TOKEN=your_token_here
     ```

5. **Run the main program:**
   ```bash
   python main.py
   ```

## 📌 Next Steps 

- 🎼 Feature extraction with `librosa`  
- 🔐 Speaker identification using `SpeechBrain`
- 📊 Visualization of speaker segments
