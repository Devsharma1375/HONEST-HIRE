from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import base64
import numpy as np
from PyPDF2 import PdfReader
import google.generativeai as genai
import os
import speech_recognition as sr
from pocketsphinx import pocketsphinx
from threading import Timer
import io

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands()

# Set up Google Gemini API
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyBfpEF_iInGTJr9eQSsn9hTkJqUrSDUM2U')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Global variables
questions = []
current_question = 0
answers = []
ratings = []

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('process_frame')
def process_frame(data):
    try:
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb_frame)

        hands_detected = results.multi_hand_landmarks is not None
        emit('hands_detection', {'hands_detected': hands_detected})
    except Exception as e:
        emit('error', {'message': f"Error processing frame: {str(e)}"})

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
            global questions
            questions = generate_questions(text)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_questions(text):
    prompt = f"Based on the following resume, generate 5 interview questions:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text.strip().split('\n')

@socketio.on('get_question')
def get_question():
    global current_question
    if current_question < len(questions):
        emit('new_question', {'question': questions[current_question], 'number': current_question + 1})
    else:
        emit('interview_complete')

@socketio.on('submit_answer')
def submit_answer(data):
    global current_question, answers, ratings
    answer = data['answer']
    answers.append(answer)
    
    evaluation = evaluate_answer(questions[current_question], answer)
    
    current_question += 1
    emit('answer_submitted', {'evaluation': evaluation})

def evaluate_answer(question, answer):
    prompt = f"Question: {question}\nAnswer: {answer}\n\nRate this answer out of 10 and provide brief feedback."
    response = model.generate_content(prompt)
    evaluation = response.text.strip()
    
    # Extract rating (assuming it's the first number in the response)
    rating = int(next(word for word in evaluation.split() if word.isdigit()))
    ratings.append(rating)
    
    return evaluation

@socketio.on('generate_report')
def generate_report():
    prompt = "Based on the following interview questions and answers, provide a comprehensive report on the candidate's performance. Include overall assessment, strengths, areas for improvement, and specific advice for each question:\n\n"
    for q, a, r in zip(questions, answers, ratings):
        prompt += f"Question: {q}\nAnswer: {a}\nRating: {r}/10\n\n"
    
    response = model.generate_content(prompt)
    report = response.text.strip()
    emit('interview_report', {'report': report})

@socketio.on('start_recording')
def start_recording():
    def stop_recording():
        emit('stop_recording')

    timer = Timer(45, stop_recording)
    timer.start()

@socketio.on('audio_data')
def receive_audio(data):
    try:
        audio_data = base64.b64decode(data.split(',')[1])
        text = transcribe_audio(audio_data)
        emit('transcription', {'text': text})
    except Exception as e:
        emit('transcription', {'text': f"Error processing audio: {str(e)}"})

def transcribe_audio(audio_data):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(io.BytesIO(audio_data))
    
    with audio_file as source:
        audio = recognizer.record(source)
    
    try:
        # Use PocketSphinx for speech recognition
        text = recognizer.recognize_sphinx(audio)
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Error with the speech recognition service; {e}"

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)