import os
from pathlib import Path
import argparse
from typing import Optional

from speech_recognition import SpeechRecognizer
from emotion_recognition import EmotionRecognizer
from database import init_db, store_interaction

class SpeechEmotionAgent:
    def __init__(
        self,
        db_path: str,
        whisper_model: str = "base",
        emotion_model: str = "r-f/wav2vec-english-speech-emotion-recognition",
        device_id: Optional[int] = None
    ):
        """Initialize the speech emotion agent."""
        # Initialize components
        self.speech_recognizer = SpeechRecognizer(model_name=whisper_model)
        self.emotion_recognizer = EmotionRecognizer(model_name=emotion_model)
        self.db_session_factory = init_db(db_path)
        self.device_id = device_id

    def process_interaction(
        self,
        duration: float = 5.0,
        user_id: Optional[str] = None,
        save_audio: bool = False
    ) -> None:
        """Record and process a single interaction."""
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
                transcript=transcript,
                emotion_label=emotion,
                confidence_score=emotion_confidence,
                user_id=user_id,
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
        # Process a single interaction
        agent.process_interaction(
            duration=args.duration,
            user_id=args.user_id,
            save_audio=args.save_audio
        )
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main()) 