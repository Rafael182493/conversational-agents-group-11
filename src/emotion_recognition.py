from typing import Tuple, Dict
import torch
import numpy as np
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

class EmotionRecognizer:
    def __init__(
        self,
        model_name: str = "r-f/wav2vec-english-speech-emotion-recognition"
    ):
        """Initialize the emotion recognizer with a Wav2Vec2 model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(self.device)
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
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

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

    def get_available_emotions(self) -> list[str]:
        """Return list of emotions that the model can recognize."""
        return list(self.id2label.values()) 