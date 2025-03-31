import os
import json
from pathlib import Path
import argparse
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from assistant_responses import AssistantResponder 

from speech_recognition import SpeechRecognizer, print_available_devices, get_available_devices
from entity_extraction import EntityExtractor
from forgetting_model import ForgettingModel
from rag import initialize_rag
from database import (
    init_db,
    store_interaction,
    create_session,
    end_session,
    get_session_interactions,
    store_entities,
    get_interaction_entities,
    get_all_session_entities,
    update_interaction_priority,
)


class SpeechAgent:
    def __init__(
            self,
            db_path: str,
            whisper_model: str = "base",
            entity_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
            device_id: Optional[int] = None
    ):
        """Initialize the speech agent."""
        # Initialize components
        self.speech_recognizer = SpeechRecognizer(model_name=whisper_model)
        self.entity_extractor = EntityExtractor(model_name=entity_model)
        self.db_session_factory = init_db(db_path)
        self.rag = initialize_rag(db_path)
        self.memory_manager = ForgettingModel(self.db_session_factory)
        self.device_id = device_id
        self.current_session_id = None
        self.assistant_responder = AssistantResponder(self.db_session_factory)

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
        """End the current session and display summary."""
        if self.current_session_id is not None:
            end_session(self.db_session_factory, self.current_session_id)
            print(f"\nEnded session {self.current_session_id}")

            # Print session summary
            interactions = get_session_interactions(self.db_session_factory, self.current_session_id)
            all_entities = get_all_session_entities(self.db_session_factory, self.current_session_id)

            print("\nSession Summary:")
            print("-" * 50)
            for interaction in interactions:
                priority_indicator = " [IMPORTANT]" if interaction.priority else ""
                print(f"Time: {interaction.timestamp}{priority_indicator}")
                print(f"Transcript: {interaction.transcript}")

                # Get entities for this specific interaction
                interaction_entities = [e for e in all_entities if e.interaction_id == interaction.id]
                if interaction_entities:
                    print("\nExtracted Entities:")
                    entity_by_type = {}
                    for entity in interaction_entities:
                        if entity.entity_type not in entity_by_type:
                            entity_by_type[entity.entity_type] = []
                        entity_by_type[entity.entity_type].append(entity.entity_value)

                    for entity_type, values in entity_by_type.items():
                        if len(values) == 1:
                            print(f"  {entity_type}: {values[0]}")
                        else:
                            print(f"  {entity_type}: {values}")

                print("-" * 50)

            pruned = self.memory_manager.forget_old_memories()
            print("pruned", pruned, "memories at the end of the session :)")
            self.current_session_id = None

    def extract_and_store_entities(self, transcript: str, interaction_id: int) -> Tuple[Dict[str, Any], bool]:
        """Extract entities from transcript and store them in the database."""
        if not transcript or transcript == "transcription failed" or interaction_id is None:
            return {}, False

        try:
            # Extract entities
            entities = self.entity_extractor.extract_entities(transcript)
            has_important_info = self._has_important_entities(entities)

            if entities:
                # Store entities in database
                stored_entities = store_entities(
                    self.db_session_factory,
                    interaction_id=interaction_id,
                    entities=entities
                )

                print(f"\nStored {len(stored_entities)} entities")
                print("Extracted entities:")
                print(json.dumps(entities, indent=2))

                if has_important_info:
                    print("Contains important planning information!")

            return entities, has_important_info
        except Exception as e:
            print(f"Error extracting or storing entities: {str(e)}")
            return {}, False

    def process_interaction(
            self,
            duration: float = 5.0,
            save_audio: bool = False,
            audio_file: str = None,
            continuous: bool = False
    ) -> None:
        """Record and process a single interaction."""
        if self.current_session_id is None:
            raise RuntimeError("No active session. Call start_session() first.")

        try:
            # Get audio data from file or trough mic
            if audio_file:
                if not os.path.isfile(audio_file):
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")

                print("Loading audio...")
                audio_file = os.path.normpath(audio_file)
                audio_data, audio_duration = self.speech_recognizer.load_audio(audio_file)

                # Process file in chunks for more realistic experience
                if continuous:
                    self._process_in_chunks(audio_data, duration, save_audio)
                # or all at once
                else:
                    self._process_audio_segment(audio_data, audio_duration, save_audio)
            else:
                # Record and process audio
                print("Listening...")
                audio_data, actual_duration = self.speech_recognizer.record_audio(
                    duration=duration,
                    device=self.device_id
                )
                self._process_audio_segment(audio_data, actual_duration, save_audio)
        except Exception as e:
            print(f"Error processing interaction: {str(e)}")
            raise

    def _process_in_chunks(self, audio_data: np.ndarray, chunk_duration: float, save_audio: bool) -> None:
        """Process audio data in chunks."""
        chunk_samples = int(chunk_duration * self.speech_recognizer.sample_rate)
        num_chunks = int(np.ceil(len(audio_data) / chunk_samples))
        full_duration = len(audio_data) / self.speech_recognizer.sample_rate

        print(f"Processing {num_chunks} chunks from {full_duration:.2f} seconds of audio...")

        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, len(audio_data))
            audio_chunk = audio_data[start_idx:end_idx]
            chunk_duration = len(audio_chunk) / self.speech_recognizer.sample_rate

            print(f"\nProcessing chunk {i + 1}/{num_chunks} ({chunk_duration:.2f} seconds)...")
            self._process_audio_segment(audio_chunk, chunk_duration, save_audio, i + 1)

    def _process_audio_segment(self, audio: np.ndarray, duration: float, save_audio: bool,
                               chunk_number: Optional[int] = None) -> None:
        """Process a segment of audio."""
        # Skip silent audio
        if np.all(audio == 0) or np.allclose(audio, 0, atol=1e-7):
            chunk_info = f" in chunk {chunk_number}" if chunk_number else ""
            print(f"\nNo audio detected{chunk_info}. Skipping.")
            return

        # Save audio if requested
        if save_audio:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_suffix = f"_chunk{chunk_number}" if chunk_number is not None else ""
            audio_dir = Path("data/audio")
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"recording_{timestamp}{chunk_suffix}.wav"
            self.speech_recognizer.save_audio(audio, str(audio_path))
            print(f"Audio saved to: {audio_path}")

        # Transcribe audio
        try:
            transcript, transcript_confidence = self.speech_recognizer.transcribe(audio)
            print(f"\nTranscript: {transcript}")
            print(f"Transcript confidence: {transcript_confidence:.2f}")
        except Exception as e:
            print(f"\nError in transcription: {str(e)}")
            transcript, transcript_confidence = "transcription failed", 0.0

        if transcript != "transcription failed":
            # 1) Store the user interaction
            user_interaction = store_interaction(
                self.db_session_factory,
                session_id=self.current_session_id,
                transcript=transcript,
                audio_duration=duration,
                role="user"  # <- We explicitly say it's from the user
            )

            # 2) Extract and store entities from the user’s transcript
            _, important_entities = self.extract_and_store_entities(transcript, user_interaction.id)

            # Possibly mark user message as priority if it has important entities
            if important_entities:
                update_interaction_priority(self.db_session_factory, user_interaction.id, True)
                print("Marked as important interaction!")

            print(f"\nStored interaction with ID: {user_interaction.id}")

            # 3) Generate the assistant's response using the entire conversation so far
            assistant_text = self.assistant_responder.get_response(self.current_session_id)


            # 4) Store the assistant response as a new interaction with role="assistant"
            assistant_interaction = store_interaction(
                self.db_session_factory,
                session_id=self.current_session_id,
                transcript=assistant_text,
                audio_duration=0.0,  # or None
                role="assistant"
            )

            # 5) Print the assistant’s response to the console
            print(f"\nAgent: {assistant_text}")

            self.assistant_responder.speak_response(assistant_text)

        else:
            print("\nSkipping storage due to failed transcription")

        # Add the rag
        if self.rag and transcript != "transcription failed":
            self.rag.store_interaction_embedding(
                self.current_session_id,
                user_interaction.id,
                transcript
            )

    def _has_important_entities(self, entities: Dict[str, Any]) -> bool:
        """Check if the extracted entities contain important information."""
        important_types = {"date", "time", "budget", "location", "venue", "deadline", "people",
                           "contact", "phone", "email", "cost", "price", "attendees", "amount"}

        return any(entity_type in important_types for entity_type in entities.keys())

    def get_memory_context(self, query: str, limit: int = 3) -> str:
        """Get relevant memory context for a given query."""
        if not self.rag:
            return "Memory retrieval not available."
        return self.rag.get_context_for_query(query, limit)


def main():
    parser = argparse.ArgumentParser(description="Speech Recognition Agent")
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
    parser.add_argument(
        "--entity-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model to use for entity extraction"
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
    agent = SpeechAgent(
        db_path=args.db_path,
        whisper_model=args.whisper_model,
        entity_model=args.entity_model,
        device_id=args.device_id
    )

    try:
        # Start a new session
        agent.start_session(user_id=args.user_id)

        if args.continuous and not args.audio_file:
            print("\nRunning in continuous mode. Press Ctrl+C to stop.")
            while True:
                agent.process_interaction(
                    duration=args.duration,
                    save_audio=args.save_audio
                )
        else:
            # Process a single interaction or perhaps an audio file
            agent.process_interaction(
                duration=args.duration,
                save_audio=args.save_audio,
                audio_file=args.audio_file,
                continuous=args.continuous
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