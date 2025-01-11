from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
from openai import OpenAI
import os
from scipy.io import wavfile
from PyPDF2 import PdfReader
import logging
from pydub import AudioSegment
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
elevenlabs_api = os.getenv('elevenlabs')
openai_api_key = os.getenv('open_ai')
elevenlabs_client = ElevenLabs(api_key=elevenlabs_api)

class AudioProcessor:
    def __init__(self):
        logger.info("Initializing AudioProcessor...")
        self.chat_history = []
        
        # Initialize models
        logger.info("Loading Whisper models...")
        self.processor = AutoProcessor.from_pretrained(
            "vinai/PhoWhisper-small",
            local_files_only=False
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "vinai/PhoWhisper-small",
            local_files_only=False
        )
        
        # Initialize chat with PDF content
        self._initialize_chat()
        logger.info("AudioProcessor initialization complete")

    def convert_audio_to_wav(self, input_path, output_path):
        """Convert audio file to WAV format with specific parameters for Whisper"""
        try:
            logger.info(f"Converting audio file from {input_path} to {output_path}")
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono and set sample rate to 16kHz
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            
            # Export as WAV
            audio.export(output_path, format="wav")
            logger.info("Audio conversion successful")
            return True
        except Exception as e:
            logger.error(f"Error converting audio: {str(e)}")
            return False

    def _read_pdf_content(self, pdf_path):
        logger.info(f"Reading PDF content from: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _initialize_chat(self):
        logger.info("Initializing chat with PDF content...")
        client = OpenAI(api_key=openai_api_key)
        pdf_path = "/Users/masterh/Desktop/WORK /Coding/viet-ttsts-local/viet-ttsts/TT KHACH THUE_ 174TH.pdf"
        pdf_content = self._read_pdf_content(pdf_path)
        self.chat_history = [
            {"role": "system", "content": f"Bạn là một người giúp đỡ trên tổng đài rất chu đáo, Nên nhớ người gọi muốn có thông tin xúc tích nên hãy trả lời các câu hỏi bằng tiếng Việt một cách ngắn gọn nhưng đầy đủ thông tin. Nếu người gọi hỏi về các câu hỏi không liên quan đến nội dung cần nhớ thì hãy trả lời rằng bạn không biết. Đây là nội dung cần nhớ: {pdf_content}"},
        ]

    def process_with_openai(self, text):
        logger.info(f"Processing with OpenAI: {text}")
        client = OpenAI(api_key=openai_api_key)
        
        self.chat_history.append({"role": "user", "content": text})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.chat_history
        )
        
        assistant_response = response.choices[0].message.content
        logger.info(f"OpenAI response: {assistant_response}")
        self.chat_history.append({"role": "assistant", "content": assistant_response})
        
        return assistant_response

    def process_audio_file(self, audio_file):
        """Process uploaded audio file and return response"""
        logger.info("Processing uploaded audio file")
        temp_input = None
        temp_wav = None
        
        try:
            # Create temporary files
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            # Save uploaded file
            audio_file.save(temp_input.name)
            logger.info(f"Saved uploaded file to {temp_input.name}")
            
            # Convert to WAV
            if not self.convert_audio_to_wav(temp_input.name, temp_wav.name):
                raise Exception("Failed to convert audio file")
            
            # Process audio with Whisper
            logger.info("Processing audio with Whisper...")
            audio_input, sampling_rate = sf.read(temp_wav.name)
            input_features = self.processor(
                audio_input,
                sampling_rate=sampling_rate,
                language="vi",
                return_tensors="pt",
                attention_mask=True
            ).input_features

            predicted_ids = self.model.generate(input_features)
            transcribed_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logger.info(f"Transcribed text: {transcribed_text}")

            # Process with OpenAI
            logger.info("Processing with OpenAI...")
            vietnamese_response = self.process_with_openai(transcribed_text)
            logger.info(f"Vietnamese response: {vietnamese_response}")

            # Generate speech with ElevenLabs
            logger.info("Generating speech with ElevenLabs...")
            audio_stream = elevenlabs_client.text_to_speech.convert_as_stream(
                text=vietnamese_response,
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Using Rachel voice ID
                model_id="eleven_flash_v2_5"
            )

            # Save response audio
            response_path = "static/response.mp3"
            os.makedirs("static", exist_ok=True)
            
            logger.info("Collecting audio stream chunks...")
            audio_bytes = b''.join(chunk for chunk in audio_stream if isinstance(chunk, bytes))
            
            with open(response_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"Saved response audio to {response_path}")

            return {
                "transcribed_text": transcribed_text,
                "response_text": vietnamese_response,
                "audio_path": response_path
            }

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}", exc_info=True)
            return {"error": str(e)}
        finally:
            # Clean up temporary files
            for temp_file in [temp_input, temp_wav]:
                if temp_file and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                        logger.info(f"Cleaned up temporary file: {temp_file.name}")
                    except Exception as e:
                        logger.error(f"Error cleaning up temporary file: {str(e)}")

    def reset_chat(self):
        """Reset the chat history and clean up any resources"""
        logger.info("Resetting chat history")
        self.chat_history = []
        # Re-initialize with system prompt
        self._initialize_chat() 