import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import whisper
import sounddevice as sd
import soundfile as sf
from scipy import signal

class SpeechRecognizer:
    def __init__(self, model_name: str = "base"):
        """Initialize the speech recognizer with a Whisper model."""
        # Whisper models are automatically cached by the library in ~/.cache/whisper
        print(f"Loading Whisper model '{model_name}'...")
        self.model = whisper.load_model(model_name)
        self.sample_rate = 16000  # Whisper expects 16kHz

    def record_audio(
        self, duration: float = 5.0, device: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """Record audio from microphone."""
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            device=device
        )
        sd.wait()
        return audio.flatten(), duration

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        """Load audio from file and resample if necessary."""
        audio, sr = sf.read(file_path)
        duration = len(audio) / sr

        if sr != self.sample_rate:
            # Resample to 16kHz
            audio = signal.resample(
                audio, int(len(audio) * self.sample_rate / sr)
            )

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono

        return audio, duration

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply preprocessing to the audio signal."""
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Apply simple noise reduction (you might want to use more sophisticated methods)
        audio = signal.medfilt(audio, kernel_size=3)
        
        return audio

    def transcribe(
        self, audio: np.ndarray, language: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Transcribe audio using Whisper.
        Returns transcript and confidence score.
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio)
        
        # Get transcription
        result = self.model.transcribe(
            audio,
            language=language,
            fp16=False  # Use float32 for better compatibility
        )
        
        transcript = result["text"].strip()
        # Average confidence across segments
        confidence = np.mean([segment["confidence"] for segment in result["segments"]])
        
        return transcript, confidence

    def save_audio(self, audio: np.ndarray, file_path: str) -> None:
        """Save audio to file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        sf.write(file_path, audio, self.sample_rate)

def get_available_devices() -> list[dict]:
    """Get list of available audio input devices."""
    return sd.query_devices() 