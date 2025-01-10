from elevenlabs import play
from elevenlabs.client import ElevenLabs
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
from pynput import keyboard
import wave
import tempfile
import soundfile as sf  # Add this import for audio file handling
from openai import OpenAI
import os
import datetime
from scipy.io import wavfile
from PyPDF2 import PdfReader  # Add this import
from threading import Event

# Load environment variables
load_dotenv()
elevenlabs_api = os.getenv('elevenlabs')
openai_api_key = os.getenv('open_ai')
elevenlabs_client = ElevenLabs(api_key=elevenlabs_api)

# Add function to read PDF content
def read_pdf_content(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Initialize chat history
chat_history = []

def process_with_openai(text, initialize=False):
    print("\nProcessing with OpenAI:", text)
    client = OpenAI(api_key=openai_api_key)
    
    if initialize:
        # Read PDF content and set up initial context
        pdf_path = "/Users/masterh/Desktop/WORK /Coding/viet-ttsts-local/viet-ttsts/TT KHACH THUE_ 174TH.pdf"
        pdf_content = read_pdf_content(pdf_path)
        chat_history.clear()
        chat_history.extend([
            {"role": "system", "content": f"Bạn là một người giúp đỡ rất chu đáo, hãy trả lời các câu hỏi bằng tiếng Việt một cách ngắn gọn nhưng đầy đủ thông tin. Đây là nội dung cần nhớ: {pdf_content}"},
        ])
        return "Chat initialized with PDF content"
    
    # Add the new user message to chat history
    chat_history.append({"role": "user", "content": text})
    
    # Get response using full chat history
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=chat_history
    )
    
    # Add assistant's response to chat history
    chat_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

def record_and_process(processor, model, stop_event=None):
    if stop_event is None:
        stop_event = Event()
    
    # Initialize chat with PDF content
    process_with_openai("Initialize chat", initialize=True)
    
    # Audio parameters
    sample_rate = 16000
    channels = 1
    
    print("Press 'k' to start recording, press 'k' again to stop. Press Ctrl+C to exit.")
    
    try:
        while not stop_event.is_set():
            recording = False
            audio_data = []
            should_process = False
            
            def on_press(key):
                nonlocal recording, should_process
                if key == keyboard.KeyCode.from_char('k'):
                    if not recording:
                        recording = True
                        print("\nRecording started... (Press 'k' again to stop)")
                    else:
                        recording = False
                        should_process = True
                        print("\nRecording stopped.")
                        return False
            
            with keyboard.Listener(on_press=on_press) as listener:
                with sd.InputStream(samplerate=sample_rate, channels=channels) as stream:
                    while listener.running:
                        if recording:
                            audio_chunk, _ = stream.read(sample_rate)
                            audio_data.extend(audio_chunk)
            
            if should_process and len(audio_data) > 0:
                print("\nSaving and processing recording...")
                
                # Convert to numpy array and scale properly
                audio_array = np.array(audio_data)
                audio_array = (audio_array * 32767).astype(np.int16)  # Scale to 16-bit PCM
                
                # Use a fixed temp file name
                temp_file_path = "temp_recording.wav"
                print(f"\nSaving to: {temp_file_path}")
                
                # Save using scipy.io.wavfile for better audio quality
                wavfile.write(temp_file_path, sample_rate, audio_array)
                
                try:
                    audio_input, sampling_rate = sf.read(temp_file_path)
                    input_features = processor(
                        audio_input,
                        sampling_rate=sampling_rate,
                        language="vi",
                        return_tensors="pt",
                        attention_mask=True
                    ).input_features
                    
                    predicted_ids = model.generate(input_features)
                    transcribed_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    print("\nTranscribed text:", transcribed_text)
                    
                    # Process with OpenAI
                    print("\nProcessing with OpenAI...")
                    vietnamese_response = process_with_openai(transcribed_text)
                    print("\nVietnamese response:", vietnamese_response)
                    
                    # Generate and play speech
                    print("\nGenerating speech with ElevenLabs...")
                    audio = elevenlabs_client.generate(
                        text=vietnamese_response,
                        voice="Rachel",
                        model="eleven_flash_v2_5"
                    )
                    print("\nPlaying audio response...")
                    play(audio)
                    
                except Exception as e:
                    print(f"\nError during processing: {str(e)}")
                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            
            print("\nPress 'k' to record again. Press Ctrl+C to exit.\n")
                
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    # Move model initialization here
    processor = AutoProcessor.from_pretrained(
        "vinai/PhoWhisper-small",
        local_files_only=False
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "vinai/PhoWhisper-small",
        local_files_only=False
    )
    
    record_and_process(processor, model)  # Pass processor and model as arguments 