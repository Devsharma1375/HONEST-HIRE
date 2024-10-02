import os
import base64
import time
import cv2
import numpy as np
import mediapipe as mp
from threading import Lock
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PyPDF2 import PdfReader
import google.generativeai as genai
import assemblyai as aai

app = Flask(__name__)
socketio = SocketIO(app)

# Set up Google Gemini API
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyBfpEF_iInGTJr9eQSsn9hTkJqUrSDUM2U')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Set up AssemblyAI API
aai.settings.api_key = "d051b8e589764ef9841d4d0ba65f9b5a"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global variables
question = ""
recording_start_time = None
hands_not_detected_start_time = None
hands_detection_buffer = []
frame_process_lock = Lock()
cheating_warnings = 0
MAX_CHEATING_WARNINGS = 3

@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        try:
            text = extract_text_from_pdf(file)
            global question
            question = generate_question(text)
            return jsonify({'success': True, 'question': question})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_question(text):
    prompt = f"Based on the following resume, generate 1 challenging interview question:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text.strip()

@socketio.on('start_recording')
def start_recording():
    global recording_start_time, hands_not_detected_start_time, cheating_warnings
    recording_start_time = time.time()
    hands_not_detected_start_time = None
    cheating_warnings = 0
    emit('recording_started')

@socketio.on('stop_recording')
def stop_recording():
    global recording_start_time, hands_not_detected_start_time
    recording_start_time = None
    hands_not_detected_start_time = None
    emit('recording_stopped')

@socketio.on('submit_audio')
def handle_audio(data):
    audio_data = base64.b64decode(data['audio'].split(',')[1])
    
    with open('audio.webm', 'wb') as f:
        f.write(audio_data)
    
    emit('progress_update', {'progress': 25})
    
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe('audio.webm')

    emit('progress_update', {'progress': 50})

    if transcript.status == aai.TranscriptStatus.error:
        emit('evaluation_result', {'error': f"Transcription Error: {transcript.error}"})
    else:
        transcription = transcript.text
        emit('progress_update', {'progress': 75})
        
        evaluation = evaluate_answer(question, transcription)
        emit('progress_update', {'progress': 100})
        
        emit('evaluation_result', {'transcription': transcription, 'evaluation': evaluation})

def evaluate_answer(question, answer):
    prompt = f"""
    Question: {question}
    Answer: {answer}

    Please provide a strict and detailed evaluation of this answer with the following:
    1. A score out of 10 (be critical and don't inflate scores)
    2. A detailed review of the answer (2-3 sentences)
    3. Three specific suggestions for improvement
    4. A list of any potential red flags (e.g., off-topic responses, vague answers, factual errors)

    Format your response as follows:
    Score: [score]/10
    Review: [your detailed review]
    Improvements:
    1. [first suggestion]
    2. [second suggestion]
    3. [third suggestion]
    Red Flags: [list any red flags, or "None" if there are no red flags]
    """
    response = model.generate_content(prompt)
    return response.text.strip()

@socketio.on('process_frame')
def process_frame(data):
    global hands_not_detected_start_time, hands_detection_buffer, recording_start_time, cheating_warnings

    with frame_process_lock:
        try:
            # Decode the image
            image_data = base64.b64decode(data['image'].split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Process the frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb_frame)

            hands_detected = results.multi_hand_landmarks is not None

            # Update detection buffer
            hands_detection_buffer.append(hands_detected)
            if len(hands_detection_buffer) > 10:
                hands_detection_buffer.pop(0)

            # Consider hands detected if they were detected in at least 6 of the last 10 frames
            hands_detected_overall = sum(hands_detection_buffer) >= 6

            current_time = time.time()

            if hands_detected_overall:
                hands_not_detected_start_time = None
            elif hands_not_detected_start_time is None:
                hands_not_detected_start_time = current_time
            elif current_time - hands_not_detected_start_time > 5:
                cheating_warnings += 1
                emit('hands_warning', {
                    'message': f'Warning {cheating_warnings}/{MAX_CHEATING_WARNINGS}: Please keep your hands visible.',
                    'warning_count': cheating_warnings
                })
                if cheating_warnings >= MAX_CHEATING_WARNINGS:
                    emit('stop_recording', {'reason': 'Multiple violations detected'})
                    recording_start_time = None
                hands_not_detected_start_time = None

            if recording_start_time and current_time - recording_start_time > 120:
                emit('stop_recording', {'reason': 'Recording time limit reached'})
                recording_start_time = None

            # Check for potential cheating behaviors
            if hands_detected_overall and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check if hand is near the edge of the frame
                    if any(landmark.x < 0.1 or landmark.x > 0.9 or landmark.y < 0.1 or landmark.y > 0.9 
                           for landmark in hand_landmarks.landmark):
                        cheating_warnings += 1
                        emit('hands_warning', {
                            'message': f'Warning {cheating_warnings}/{MAX_CHEATING_WARNINGS}: Keep hands within the frame.',
                            'warning_count': cheating_warnings
                        })
                        if cheating_warnings >= MAX_CHEATING_WARNINGS:
                            emit('stop_recording', {'reason': 'Multiple violations detected'})
                            recording_start_time = None
                        break

            emit('hands_detection', {'hands_detected': hands_detected_overall})

        except Exception as e:
            emit('error', {'message': f"Error processing frame: {str(e)}"})

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)