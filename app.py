from flask import Flask, render_template, request, jsonify
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from dotenv import load_dotenv
import soundfile as sf
import numpy as np
import wave
import tempfile
import openai
import os
import base64
import io

app = Flask(__name__)

# Load environment variables
load_dotenv()
elevenlabs_api = os.getenv('elevenlabs')
openai.api_key = os.getenv('open_ai')

# Initialize models
processor = AutoProcessor.from_pretrained("vinai/PhoWhisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("vinai/PhoWhisper-small")
elevenlabs_client = ElevenLabs(api_key=elevenlabs_api)

def process_with_openai(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Bạn là một người giúp đỡ rất chu đáo, hãy trả lời các câu hỏi bằng tiếng Việt"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Get audio data from the request
        audio_data = request.json['audio']
        audio_binary = base64.b64decode(audio_data.split(',')[1])
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_binary)
            temp_file_path = temp_file.name

        # Process audio with Whisper
        audio_input, sampling_rate = sf.read(temp_file_path)
        input_features = processor(
            audio_input,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features
        
        predicted_ids = model.generate(input_features)
        transcribed_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Process with OpenAI
        vietnamese_response = process_with_openai(transcribed_text)
        
        # Generate speech with ElevenLabs
        audio = elevenlabs_client.generate(
            text=vietnamese_response,
            voice="Nicole",
            model="eleven_multilingual_v2"
        )
        play(audio)
        
        # Clean up
        os.unlink(temp_file_path)
        
        return jsonify({
            'transcribed_text': transcribed_text,
            'response': vietnamese_response,
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 