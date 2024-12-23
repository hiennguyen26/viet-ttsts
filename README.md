# Voice Chat Interface

A Flask web application that provides a voice chat interface with speech-to-text, OpenAI processing, and text-to-speech capabilities.

## Setup

1. Create a `.env` file in the root directory with your API keys:
```
elevenlabs=your_elevenlabs_api_key
open_ai=your_openai_api_key
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Hold the SPACE key to start recording your voice
2. Release the SPACE key to stop recording and process your speech
3. Wait for the response to be processed and played through your speakers
4. The spacebar will be disabled during processing to prevent overlapping requests

## Features

- Speech-to-text using VINAI's PhoWhisper model
- OpenAI processing for Vietnamese responses
- Text-to-speech using ElevenLabs
- Real-time chat interface
- Visual feedback for recording status 