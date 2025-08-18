from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yagmail
import PyPDF2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# Lazy-load model
model = None
def get_model():
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Calculate similarity between JD and Resume
def get_similarity(jd_text, resume_text):
    model = get_model()
    embeddings = model.encode([jd_text, resume_text])
    sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim_score

# Send email
def send_email(receiver_email, subject, body):
    yag = yagmail.SMTP(user=EMAIL_USER, password=EMAIL_PASS)
    yag.send(to=receiver_email, subject=subject, contents=body)

# Extract text from uploaded PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Flask App
app = Flask(__name__)
CORS(app)  # enable CORS for frontend POST requests

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        jd = request.form.get('job_desc')
        email = request.form.get('email')
        resume_file = request.files.get('resume')

        if not jd or not email or not resume_file:
            return jsonify({"error": "Missing job description, email, or resume"}), 400

        resume_text = extract_text_from_pdf(resume_file)
        similarity = get_similarity(jd, resume_text)

        if similarity >= 0.7:
            subject = "Congratulations - Shortlisted for HR Round"
            body = "You have been shortlisted based on your resume. We'll contact you soon!"
        else:
            subject = "Application Status"
            body = "Thank you for applying. Unfortunately, you're not shortlisted at this time."

        send_email(email, subject, body)
        return jsonify({"status": "success", "similarity": similarity})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check (optional for Render uptime)
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run()
