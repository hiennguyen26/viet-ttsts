from flask import Flask, render_template, jsonify, send_file, send_from_directory, request
from flask_pipeline import AudioProcessor
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure static folder exists
os.makedirs('static', exist_ok=True)

# Global variables
audio_processor = None

def init_audio_processor():
    global audio_processor
    if audio_processor is None:
        logger.info("Initializing AudioProcessor...")
        audio_processor = AudioProcessor()
        logger.info("AudioProcessor initialized")

@app.route('/')
def home():
    logger.info("Serving home page")
    init_audio_processor()
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    logger.info("Audio upload received")
    init_audio_processor()
    
    if 'audio' not in request.files:
        logger.error("No audio file in request")
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    
    try:
        result = audio_processor.process_audio_file(audio_file)
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    logger.info(f"Serving audio file: {filename}")
    try:
        return send_from_directory('static', filename, mimetype='audio/mpeg')
    except Exception as e:
        logger.error(f"Error serving audio file: {e}")
        return jsonify({"error": "Audio file not found"}), 404

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    global audio_processor
    logger.info("Chat reset requested")
    
    if audio_processor:
        try:
            # Reset the chat history
            audio_processor.reset_chat()
            # Clean up audio files
            for file in os.listdir('static'):
                if file.endswith('.mp3'):
                    os.remove(os.path.join('static', file))
            logger.info("Chat reset successful")
            return jsonify({"status": "success"})
        except Exception as e:
            logger.error(f"Error resetting chat: {str(e)}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "error", "message": "Audio processor not initialized"})

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5050, debug=True) 