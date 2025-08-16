from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yagmail
import PyPDF2
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from  .env file into environment

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# Load the SentenceTransformer model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to calculate similarity between JD and Resume
def get_similarity(jd_text, resume_text):
    embeddings = model.encode([jd_text, resume_text])
    sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim_score

# Function to send email
def send_email(receiver_email, subject, body):
    yag = yagmail.SMTP(user=EMAIL_USER, password = EMAIL_PASS)  # Use App Password
    yag.send(to=receiver_email, subject=subject, contents=body)

# Function to extract text from uploaded PDF resume
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        jd = request.form.get('job_desc')
        email = request.form.get('email')
        resume_file = request.files.get('resume')


        resume_text = extract_text_from_pdf(resume_file)
        similarity = get_similarity(jd, resume_text)

        if similarity >= 0.7:
            subject = "Congratulations - Shortlisted for HR Round"
            body = "You have been shortlisted based on your resume. We'll contact you soon!"
        else:
            subject = "Application Status"
            body = "Thank you for applying. Unfortunately, you're not shortlisted at this time."

        send_email(email, subject, body)
        # return f"Processed. Similarity: {similarity:.2f}"
        return render_template('next.html')

    return render_template('index.html')


@app.route('/done',methods=['GET','POST'])
def done():
    if request.method == 'GET':
        print('Done')
        return render_template('index.html')
        
    return render_template('next.html')



if __name__ == '__main__':
    app.run(debug=True)
