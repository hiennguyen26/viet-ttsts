from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import soundfile as sf
from openai import OpenAI
import os
from scipy.io import wavfile
from PyPDF2 import PdfReader
from threading import Event, Lock
import queue
import time
import logging

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
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self.stream_lock = Lock()
        
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
            {"role": "system", "content": f"Bạn là một người giúp đỡ rất chu đáo, hãy trả lời các câu hỏi bằng tiếng Việt một cách ngắn gọn nhưng đầy đủ thông tin. Đây là nội dung cần nhớ: {pdf_content}"},
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

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.is_recording:
            try:
                self.audio_queue.put(indata.copy())
                logger.debug(f"Added {len(indata)} samples to queue. Queue size: {self.audio_queue.qsize()}")
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")

    def start_recording(self):
        with self.stream_lock:
            if self.is_recording:
                logger.warning("Already recording!")
                return
            
            logger.info("Starting audio recording...")
            self.is_recording = True
            self.audio_queue = queue.Queue()
            
            try:
                self.stream = sd.InputStream(
                    samplerate=16000,
                    channels=1,
                    callback=self.audio_callback,
                    blocksize=1024
                )
                self.stream.start()
                logger.info("Recording stream started successfully")
            except Exception as e:
                self.is_recording = False
                logger.error(f"Failed to start recording: {e}")
                raise

    def stop_recording(self):
        with self.stream_lock:
            if not self.is_recording:
                logger.warning("Not currently recording!")
                return "Not recording"
            
            logger.info("Stopping recording...")
            self.is_recording = False
            
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None
                except Exception as e:
                    logger.error(f"Error closing stream: {e}")
            
            return self.process_recording()

    def process_recording(self):
        logger.info("Processing recording...")
        if self.audio_queue.empty():
            logger.warning("No audio recorded!")
            return "No audio recorded"

        # Combine all audio chunks
        audio_data = []
        queue_size = self.audio_queue.qsize()
        logger.info(f"Processing {queue_size} audio chunks")
        
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get()
            audio_data.append(chunk)

        if not audio_data:
            logger.warning("No audio data collected")
            return "No audio data"

        # Combine chunks and convert to numpy array
        audio_array = np.concatenate(audio_data)
        audio_array = (audio_array * 32767).astype(np.int16)
        
        logger.info(f"Final audio array shape: {audio_array.shape}")

        # Save temporary WAV file
        temp_file_path = "temp_recording.wav"
        wavfile.write(temp_file_path, 16000, audio_array)
        logger.info(f"Saved temporary recording to {temp_file_path}")

        try:
            # Process audio with Whisper
            logger.info("Processing audio with Whisper...")
            audio_input, sampling_rate = sf.read(temp_file_path)
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

            # Generate speech with ElevenLabs using new streaming API
            logger.info("Generating speech with ElevenLabs...")
            audio_stream = elevenlabs_client.text_to_speech.convert_as_stream(
                text=vietnamese_response,
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Using Rachel voice ID
                model_id="eleven_flash_v2_5"
            )

            # Collect audio chunks and save to file
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
            return str(e)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info("Cleaned up temporary recording file")

    def cleanup(self):
        logger.info("Cleaning up AudioProcessor...")
        self.is_recording = False
        with self.stream_lock:
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}") 