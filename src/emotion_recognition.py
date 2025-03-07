from typing import Tuple, Dict
import torch
import numpy as np
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)
import os

class EmotionRecognizer:
    def __init__(
        self,
        model_name: str = "wav2vec2-emotion",
        model_path: str = "models"
    ):
        """Initialize the emotion recognizer with a Wav2Vec2 model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = os.path.join(model_path, model_name)
        
        if not os.path.exists(model_dir):
            print(f"Local model not found at {model_dir}, downloading from HuggingFace...")
            model_name = "r-f/wav2vec-english-speech-emotion-recognition"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            
            # Save models locally
            os.makedirs(model_dir, exist_ok=True)
            self.feature_extractor.save_pretrained(model_dir)
            self.model.save_pretrained(model_dir)
        else:
            print(f"Loading model from {model_dir}")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
        
        self.model = self.model.to(self.device)
        self.id2label = self.model.config.id2label
        self.sample_rate = self.feature_extractor.sampling_rate

    def predict_emotion(self, audio: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict emotion from audio array.
        Returns:
            - predicted emotion label
            - confidence score
            - dictionary of all emotion probabilities
        """
        try:
            # Check if audio is completely silent or too quiet
            if np.all(audio == 0) or np.allclose(audio, 0, atol=1e-7) or np.max(np.abs(audio)) < 1e-6:
                print("Warning: Audio is silent or too quiet. Defaulting to neutral emotion.")
                return "neutral", 1.0, {"neutral": 1.0}

            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize audio if needed
            max_abs = np.max(np.abs(audio))
            if max_abs > 0:
                audio = audio / max_abs

            # Extract features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Check for NaN values
                if torch.isnan(logits).any():
                    print("Warning: NaN values detected in model output. Defaulting to neutral emotion.")
                    return "neutral", 1.0, {"neutral": 1.0}
                
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Get predicted label and confidence
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            predicted_label = self.id2label[predicted_id]
            confidence_score = probabilities[0][predicted_id].item()

            # Get all emotion probabilities
            emotion_probs = {
                self.id2label[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }

            return predicted_label, confidence_score, emotion_probs
        except Exception as e:
            print(f"Error in emotion prediction: {str(e)}")
            print("Defaulting to neutral emotion.")
            return "neutral", 1.0, {"neutral": 1.0}

    def get_available_emotions(self) -> list[str]:
        """Return list of emotions that the model can recognize."""
        return list(self.id2label.values()) 