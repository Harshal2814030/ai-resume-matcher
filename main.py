import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(tokens)

print("Paste your resume. Press Enter twice when done:\n")
resume = ""
while True:
    line = input()
    if line == "":
        break
    resume += line + " "

print("\nPaste the job description. Press Enter twice when done:\n")
job = ""
while True:
    line = input()
    if line == "":
        break
    job += line + " "

resume_clean = clean_text(resume)
job_clean = clean_text(job)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume_clean, job_clean])

similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

print(f"\n🔥 Match Score: {round(similarity*100, 2)}%")
