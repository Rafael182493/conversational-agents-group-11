import os
import tempfile
import numpy as np
import pytest
from pathlib import Path

from src.speech_recognition import SpeechRecognizer
from src.emotion_recognition import EmotionRecognizer
from src.database import init_db, store_interaction

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db') as f:
        yield f.name

@pytest.fixture
def sample_audio():
    """Generate a simple test audio signal."""
    duration = 1.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)

def test_speech_recognizer_initialization():
    """Test that speech recognizer can be initialized."""
    recognizer = SpeechRecognizer(model_name="tiny")
    assert recognizer is not None
    assert recognizer.sample_rate == 16000

def test_emotion_recognizer_initialization():
    """Test that emotion recognizer can be initialized."""
    recognizer = EmotionRecognizer()
    assert recognizer is not None
    assert len(recognizer.get_available_emotions()) > 0

def test_database_initialization(temp_db):
    """Test database initialization and basic operations."""
    # Initialize database
    session_factory = init_db(temp_db)
    assert os.path.exists(temp_db)

    # Store a test interaction
    interaction = store_interaction(
        session_factory,
        transcript="Hello, world!",
        emotion_label="neutral",
        confidence_score=0.95,
        user_id="test_user",
        audio_duration=1.0
    )
    assert interaction.id is not None
    assert interaction.transcript == "Hello, world!"
    assert interaction.emotion_label == "neutral"

def test_audio_preprocessing(sample_audio):
    """Test audio preprocessing functionality."""
    recognizer = SpeechRecognizer(model_name="tiny")
    processed_audio = recognizer.preprocess_audio(sample_audio)
    
    # Check that audio is normalized
    assert np.max(np.abs(processed_audio)) <= 1.0
    
    # Check that output shape matches input
    assert processed_audio.shape == sample_audio.shape 