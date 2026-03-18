print("--- Starting Flask App ---")

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pdfplumber
import docx
import re
from system import RAGSystem
from transformers import pipeline

# Initialize Flask app before routes
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB limit

# Initialize RAG system and QA pipeline
rag = RAGSystem()
qa_pipeline = pipeline("text-generation", model="google/flan-t5-base")

def allowed_file(filename):
    return True  # Allow all file types for now

def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    text = ""
    try:
        if ext == 'pdf':
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        elif ext == 'docx':
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = ""
    except Exception as e:
        print(f"Failed to extract text from {filepath}: {e}")
        text = ""

    # Improved chunking by paragraph and sentences
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    fine_chunks = []
    for chunk in chunks:
        fine_chunks.extend([sentence.strip() for sentence in re.split(r'\.\s*', chunk) if sentence.strip()])
    return fine_chunks

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files' not in request.files:
            return render_template('index.html', message="No file part in the request.")
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return render_template('index.html', message="No files selected.")

        all_chunks = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                text_chunks = extract_text(filepath)
                if text_chunks:
                    all_chunks.extend(text_chunks)

        if all_chunks:
            rag.load_documents(all_chunks)
            return render_template('index.html', message=f'{len(files)} file(s) uploaded and processed.')
        else:
            return render_template('index.html', message="No extractable text found in uploaded files.")
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    questions_text = request.form.get('questions')
    if not questions_text:
        return render_template('index.html', message='Please enter at least one question.')

    questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
    answers = []
    for question in questions:
        results = rag.retrieve(question, top_k=1)
        best_chunk = results[0][0] if results else ""
        if best_chunk:
            try:
                prompt = f"Context: {best_chunk}\nQuestion: {question}\nAnswer:"
                response = qa_pipeline(
    prompt,
    max_length=80,
    num_return_sequences=1,
    do_sample=False
)

                full_text = response[0]['generated_text']
                answer = ". ".join(answer.split(". ")[:2])
            except Exception as e:
                answer = f"[Model error: {e}]"
            answers.append(f"Q: {question}\nA: {answer}\nContext: {best_chunk}")
        else:
            answers.append(f"Q: {question}\nA: No relevant answer found.")

    combined_answers = "\n\n---\n\n".join(answers)
    return render_template('index.html', answer=combined_answers)

if __name__ == '__main__':
    app.run(debug=True)
