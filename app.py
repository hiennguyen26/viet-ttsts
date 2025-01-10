from flask import Flask, render_template, jsonify, send_file, send_from_directory
from flask_pipeline import AudioProcessor
import threading
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

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

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global audio_processor
    logger.info("Start recording request received")
    init_audio_processor()
    
    if audio_processor:
        try:
            audio_processor.start_recording()
            logger.info("Recording started successfully")
            return jsonify({"status": "started"})
        except Exception as e:
            logger.error(f"Error starting recording: {str(e)}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "error", "message": "Audio processor not initialized"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global audio_processor
    logger.info("Stop recording request received")
    
    if audio_processor:
        try:
            result = audio_processor.stop_recording()
            logger.info(f"Recording stopped. Result: {result}")
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error stopping recording: {str(e)}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "error", "message": "Audio processor not initialized"})

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    logger.info(f"Serving audio file: {filename}")
    try:
        return send_from_directory('static', filename, mimetype='audio/mpeg')
    except Exception as e:
        logger.error(f"Error serving audio file: {e}")
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True) 