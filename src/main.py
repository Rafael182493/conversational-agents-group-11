import os
from pathlib import Path
import argparse
from typing import Optional
from datetime import datetime

from speech_recognition import SpeechRecognizer
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
        save_audio: bool = False
    ) -> None:
        """Record and process a single interaction."""
        if self.current_session_id is None:
            raise RuntimeError("No active session. Call start_session() first.")

        try:
            # Record audio
            print("Listening...")
            audio, actual_duration = self.speech_recognizer.record_audio(
                duration=duration,
                device=self.device_id
            )

            # Save audio if requested
            if save_audio:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_dir = Path("data/audio")
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / f"recording_{timestamp}.wav"
                self.speech_recognizer.save_audio(audio, str(audio_path))
                print(f"Audio saved to: {audio_path}")

            # Get transcript
            transcript, transcript_confidence = self.speech_recognizer.transcribe(audio)
            print(f"\nTranscript: {transcript}")
            print(f"Transcript confidence: {transcript_confidence:.2f}")

            # Get emotion
            emotion, emotion_confidence, emotion_probs = self.emotion_recognizer.predict_emotion(audio)
            print(f"\nDetected emotion: {emotion}")
            print(f"Emotion confidence: {emotion_confidence:.2f}")
            print("\nEmotion probabilities:")
            for emotion_label, prob in emotion_probs.items():
                print(f"  {emotion_label}: {prob:.2f}")

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
        "--device-id",
        type=int,
        help="Audio input device ID"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run in continuous mode, processing multiple interactions"
    )

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

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
                save_audio=args.save_audio
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