import os
from pathlib import Path
import argparse
from typing import Optional
from datetime import datetime
import numpy as np

from speech_recognition import SpeechRecognizer, print_available_devices, get_available_devices
from emotion_recognition import EmotionRecognizer
from database import (
    init_db,
    store_interaction,
    create_session,
    end_session,
    get_session_interactions
)

class SpeechEmotionAgent:
    def __init__(
        self,
        db_path: str,
        whisper_model: str = "base",
        emotion_model: str = "wav2vec2-emotion",
        device_id: Optional[int] = None
    ):
        """Initialize the speech emotion agent."""
        # Initialize components
        self.speech_recognizer = SpeechRecognizer(model_name=whisper_model)
        self.emotion_recognizer = EmotionRecognizer(model_name=emotion_model)
        self.db_session_factory = init_db(db_path)
        self.device_id = device_id
        self.current_session_id = None

    def start_session(self, user_id: Optional[str] = None) -> None:
        """Start a new conversation session."""
        try:
            session = create_session(self.db_session_factory, user_id)
            self.current_session_id = session.id if session else None
            if self.current_session_id:
                print(f"\nStarted new session with ID: {self.current_session_id}")
            else:
                raise RuntimeError("Failed to create new session")
        except Exception as e:
            print(f"Error starting session: {str(e)}")
            raise

    def end_current_session(self) -> None:
        """End the current session."""
        if self.current_session_id is not None:
            end_session(self.db_session_factory, self.current_session_id)
            print(f"\nEnded session {self.current_session_id}")
            
            # Print session summary
            interactions = get_session_interactions(self.db_session_factory, self.current_session_id)
            print("\nSession Summary:")
            print("-" * 50)
            for interaction in interactions:
                print(f"Time: {interaction.timestamp}")
                print(f"Transcript: {interaction.transcript}")
                print(f"Emotion: {interaction.emotion_label} (confidence: {interaction.confidence_score:.2f})")
                print("-" * 50)
            self.current_session_id = None

    def process_interaction(
        self,
        duration: float = 5.0,
        save_audio: bool = False,
        audio_file: str = None
    ) -> None:
        """Record and process a single interaction."""
        if self.current_session_id is None:
            raise RuntimeError("No active session. Call start_session() first.")

        try:
            # Load audio
            if audio_file:
                print("Loading audio...")
                # Check if file exists
                if not os.path.isfile(audio_file):
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")
                
                # Normalize path (resolves any .. or . in the path)
                audio_file = os.path.normpath(audio_file)
                audio, actual_duration = self.speech_recognizer.load_audio(audio_file)
            else:
                # Record audio
                print("Listening...")
                audio, actual_duration = self.speech_recognizer.record_audio(
                    duration=duration,
                    device=self.device_id
                )

            # Check if audio is too quiet
            if np.all(audio == 0) or np.allclose(audio, 0, atol=1e-7):
                print("\nNo audio detected. Please try speaking louder.")
                return

            # Save audio if requested
            if save_audio:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_dir = Path("data/audio")
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / f"recording_{timestamp}.wav"
                self.speech_recognizer.save_audio(audio, str(audio_path))
                print(f"Audio saved to: {audio_path}")

            # Get transcript
            try:
                transcript, transcript_confidence = self.speech_recognizer.transcribe(audio)
                print(f"\nTranscript: {transcript}")
                print(f"Transcript confidence: {transcript_confidence:.2f}")
            except Exception as e:
                print(f"\nError in transcription: {str(e)}")
                transcript = "transcription failed"
                transcript_confidence = 0.0

            # Get emotion
            try:
                emotion, emotion_confidence, emotion_probs = self.emotion_recognizer.predict_emotion(audio)
                print(f"\nDetected emotion: {emotion}")
                print(f"Emotion confidence: {emotion_confidence:.2f}")
                print("\nEmotion probabilities:")
                for emotion_label, prob in emotion_probs.items():
                    print(f"  {emotion_label}: {prob:.2f}")
            except Exception as e:
                print(f"\nError in emotion detection: {str(e)}")
                emotion = "unknown"
                emotion_confidence = 0.0
                emotion_probs = {}

            # Only store if we have some meaningful content
            if transcript != "transcription failed" or emotion != "unknown":
                # Store in database
                interaction = store_interaction(
                    self.db_session_factory,
                    session_id=self.current_session_id,
                    transcript=transcript,
                    emotion_label=emotion,
                    confidence_score=emotion_confidence,
                    audio_duration=actual_duration
                )
                print(f"\nStored interaction with ID: {interaction.id}")
            else:
                print("\nSkipping storage due to failed processing")

        except Exception as e:
            print(f"Error processing interaction: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition Agent")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/agent_data.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model to use (tiny, base, small, medium, large)"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="Optional user ID for GDPR compliance"
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save recorded audio to file"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        help="Audio input device ID (use --list-devices to see available devices)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run in continuous mode, processing multiple interactions"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Audio file to load instead of recording"
    )

    args = parser.parse_args()

    # If --list-devices is specified, print devices and exit
    if args.list_devices:
        print_available_devices()
        return 0

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

    # Validate device ID if specified
    if args.device_id is not None:
        available_devices = get_available_devices()
        device_ids = [d['id'] for d in available_devices]
        if args.device_id not in device_ids:
            print(f"Error: Device ID {args.device_id} not found in available devices.")
            print("\nAvailable devices:")
            print_available_devices()
            return 1

    # Initialize agent
    agent = SpeechEmotionAgent(
        db_path=args.db_path,
        whisper_model=args.whisper_model,
        device_id=args.device_id
    )

    try:
        # Start a new session
        agent.start_session(user_id=args.user_id)

        if args.continuous:
            print("\nRunning in continuous mode. Press Ctrl+C to stop.")
            while True:
                agent.process_interaction(
                    duration=args.duration,
                    save_audio=args.save_audio
                )
        else:
            # Process a single interaction
            agent.process_interaction(
                duration=args.duration,
                save_audio=args.save_audio,
                audio_file=args.audio_file
            )
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        # End the session
        if agent.current_session_id is not None:
            agent.end_current_session()

    return 0

if __name__ == "__main__":
    exit(main()) 