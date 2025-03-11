import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import whisper
import sounddevice as sd
import soundfile as sf
from scipy import signal

class SpeechRecognizer:
    def __init__(self, model_name: str = "medium"):
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
        audio = audio.flatten()
        
        # Check audio levels
        audio_max = np.max(np.abs(audio))
        if audio_max < 0.01:  # Very quiet
            print("Warning: Audio level is very low")
        elif audio_max > 0.9:  # Close to clipping
            print("Warning: Audio level is very high, might be clipping")
            
        return audio, duration

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        """Load audio from file and resample if necessary."""
        audio, sr = sf.read(file_path, dtype='float32')
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
        # Check if audio is completely silent
        if np.all(audio == 0) or np.allclose(audio, 0, atol=1e-7):
            print("Warning: Detected silent audio")
            return np.zeros_like(audio)
        
        # Normalize audio (safely)
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs
        else:
            return np.zeros_like(audio)
        
        # Apply noise reduction
        # Use a larger kernel size for better noise reduction
        audio = signal.medfilt(audio, kernel_size=5)
        
        # Apply a simple noise gate
        noise_gate = 0.01
        audio[np.abs(audio) < noise_gate] = 0
        
        return audio

    def transcribe(
        self, audio: np.ndarray, language: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Transcribe audio using Whisper.
        Returns transcript and confidence score.
        """
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio)
            
            # Skip transcription if audio is silent
            if np.all(audio == 0) or np.allclose(audio, 0, atol=1e-7):
                return "", 0.0
            
            # Get transcription
            result = self.model.transcribe(
                audio,
                language=language,
                fp16=False  # Use float32 for better compatibility
            )
            
            transcript = result["text"].strip()
            
            # Handle empty transcript
            if not transcript:
                return "", 0.0
                
            # Safely calculate confidence
            confidences = [segment.get("confidence", 0.0) for segment in result["segments"]]
            confidence = np.mean(confidences) if confidences else 0.0
            
            return transcript, confidence
            
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return "", 0.0

    def save_audio(self, audio: np.ndarray, file_path: str) -> None:
        """Save audio to file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        sf.write(file_path, audio, self.sample_rate)

def get_available_devices() -> list[dict]:
    """Get list of available audio input devices.
    
    Returns:
        list[dict]: List of dictionaries containing device information:
            - id: Device index
            - name: Device name
            - channels: Number of input channels
            - default: Whether this is the default input device
            - samplerate: Default samplerate
    """
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        # Only include devices with input channels
        if device['max_input_channels'] > 0:
            input_devices.append({
                'id': i,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'default': i == sd.default.device[0],
                'samplerate': device['default_samplerate']
            })
    
    return input_devices

def print_available_devices() -> None:
    """Print all available input devices in a formatted way."""
    devices = get_available_devices()
    
    print("\nAvailable Input Devices:")
    print("-" * 60)
    for device in devices:
        default_marker = " (Default)" if device['default'] else ""
        print(f"Device ID: {device['id']}{default_marker}")
        print(f"Name: {device['name']}")
        print(f"Input Channels: {device['channels']}")
        print(f"Sample Rate: {device['samplerate']} Hz")
        print("-" * 60) 