import os
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PyPDF2 import PdfReader
import google.generativeai as genai
import assemblyai as aai
import base64
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Set up Google Gemini API
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyBfpEF_iInGTJr9eQSsn9hTkJqUrSDUM2U')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Set up AssemblyAI API
aai.settings.api_key = "d051b8e589764ef9841d4d0ba65f9b5a"

question = ""

@app.route('/')
def index():
    return render_template('index3.html')

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

@socketio.on('submit_audio')
def handle_audio(data):
    audio_data = base64.b64decode(data['audio'].split(',')[1])
    
    with open('audio.webm', 'wb') as f:
        f.write(audio_data)
    
    # Simulate progress updates
    emit('progress_update', {'progress': 25})
    time.sleep(1)  # Simulate processing time
    
    # Transcribe the audio using AssemblyAI
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe('audio.webm')

    emit('progress_update', {'progress': 50})
    time.sleep(1)  # Simulate processing time

    if transcript.status == aai.TranscriptStatus.error:
        emit('evaluation_result', {'error': f"Transcription Error: {transcript.error}"})
    else:
        transcription = transcript.text
        emit('progress_update', {'progress': 75})
        time.sleep(1)  # Simulate processing time
        
        evaluation = evaluate_answer(question, transcription)
        emit('progress_update', {'progress': 100})
        time.sleep(0.5)  # Short delay before sending final result
        
        emit('evaluation_result', {'transcription': transcription, 'evaluation': evaluation})

def evaluate_answer(question, answer):
    prompt = f"""
    Question: {question}
    Answer: {answer}

    Please provide a brief evaluation of this answer with the following:
    1. A score out of 10
    2. A brief review of the answer (1-2 sentences)
    3. One suggestion for improvement

    Format your response as follows:
    Score: [score]/10
    Review: [your review]
    Improvement: [your suggestion]
    """
    response = model.generate_content(prompt)
    return response.text.strip()

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)