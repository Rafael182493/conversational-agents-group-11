# Speech Emotion Recognition Agent

A privacy-focused speech recognition system that transcribes spoken input and detects emotions from speech, storing the results locally in a SQLite database.

## Features

- **Speech Recognition**: Uses OpenAI's Whisper model for accurate speech-to-text conversion
- **Emotion Detection**: Employs Wav2Vec2-based model for speech emotion recognition
- **Local Processing**: All processing happens on your machine, ensuring privacy
- **Data Storage**: Stores transcripts and emotions in a SQLite database
- **GDPR Compliant**: Supports user data management and deletion
- **Dockerized**: Easy deployment with Docker

## Requirements

- Python 3.10+
- Docker (optional, for containerized deployment)
- PortAudio (for microphone access)
- FFmpeg (for audio processing)

## Installation

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/speech-emotion-agent.git
   cd speech-emotion-agent
   ```

2. Install system dependencies (Ubuntu/Debian):
   ```bash
   sudo apt-get update
   sudo apt-get install -y portaudio19-dev python3-pyaudio ffmpeg
   ```

3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t speech-emotion-agent .
   ```

## Usage

### Local Usage

Run the agent with default settings:
```bash
python src/main.py
```

Available command-line arguments:
- `--db-path`: Path to SQLite database (default: "data/agent_data.db")
- `--duration`: Recording duration in seconds (default: 5.0)
- `--whisper-model`: Whisper model size (tiny/base/small/medium/large)
- `--user-id`: Optional user ID for GDPR compliance
- `--save-audio`: Save recorded audio to file
- `--device-id`: Audio input device ID

### Docker Usage

Run the container with default settings:
```bash
docker run -it --rm \
  --device /dev/snd:/dev/snd \
  -v $(pwd)/data:/app/data \
  speech-emotion-agent
```

Override default arguments:
```bash
docker run -it --rm \
  --device /dev/snd:/dev/snd \
  -v $(pwd)/data:/app/data \
  speech-emotion-agent \
  --duration 10.0 \
  --whisper-model small \
  --save-audio
```

## Data Management

The system stores all data in a SQLite database with the following schema:

```sql
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    transcript TEXT NOT NULL,
    emotion_label TEXT NOT NULL,
    confidence_score FLOAT,
    user_id TEXT,
    audio_duration FLOAT
);
```

### GDPR Compliance

To delete all data associated with a user:
```python
from database import init_db, delete_user_data
session_factory = init_db("data/agent_data.db")
delete_user_data(session_factory, "user123")
```

## Models

- **Speech Recognition**: OpenAI's Whisper (default: "base" model)
- **Emotion Recognition**: Wav2Vec2-based model from HuggingFace

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI's Whisper for speech recognition
- HuggingFace's Transformers library
- The Wav2Vec2 community for emotion recognition models 