from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from docx import Document
import re
import io
import os

# --- Synonyms dictionary ---
SYNONYMS = {
    "rest api": ["api", "restapi", "rest apis"],
    "ci/cd": ["ci cd", "continuous integration", "continuous deployment"],
    "mlops": ["machine learning ops", "ml ops"],
    "deep learning": ["dl"],
    "computer vision": ["cv"],
    "data visualization": ["visualization", "data viz"]
}

def extract_text_from_pdf(file_path_or_bytes):
    text = ""
    pdf = pdfplumber.open(file_path_or_bytes)
    with pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path_or_bytes):
    doc = Document(file_path_or_bytes)
    return "\n".join([p.text for p in doc.paragraphs])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9+/.\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_keywords_from_jd(jd_text):
    common_terms = [
        "java", "python", "c++", "sql", "nosql", "aws", "azure", "gcp",
        "docker", "kubernetes", "rest api", "microservices", "ci/cd",
        "machine learning", "deep learning", "nlp", "pytorch", "tensorflow",
        "data structures", "algorithms", "system design"
    ]
    return list({term for term in common_terms if term in jd_text})

def keyword_match_score_loose(resume_text, skills):
    matches = 0
    for skill in skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', resume_text):
            matches += 1
            continue
        if skill in SYNONYMS:
            if any(re.search(r'\b' + re.escape(syn) + r'\b', resume_text) for syn in SYNONYMS[skill]):
                matches += 1
    return (matches / len(skills)) * 100 if skills else 0

def tfidf_score_boost(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

def final_score(kw_score, tfidf_score):
    return 0.5 * kw_score + 0.5 * tfidf_score

# === FastAPI App with API Key ===
API_KEY = "test123"
app = FastAPI(title="ATS Resume Checker API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve Frontend ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")  # ✅ frontend folder inside project

# Mount static assets (CSS/JS if any)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.get("/health")
def health():
    return {"status": "ok"}

def verify_api_key(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ---------- Shared analysis ----------
def analyze_resume(resume_text: str, jd_text: str):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    jd_keywords = extract_keywords_from_jd(jd_clean)
    kw_score = keyword_match_score_loose(resume_clean, jd_keywords)
    tfidf = tfidf_score_boost(resume_clean, jd_clean)
    final = final_score(kw_score, tfidf)
    missing_skills = [s for s in jd_keywords if not re.search(r'\b' + re.escape(s) + r'\b', resume_clean)]

    return {
        "keyword_match_score": round(kw_score, 2),
        "tfidf_score": round(tfidf, 2),
        "final_score": round(final, 2),
        "jd_keywords": jd_keywords,
        "missing_skills": missing_skills
    }

# ---------- 1) Multipart: file + jd_text ----------
@app.post("/check_resume")
async def check_resume(
    api_key: str = Header(..., alias="X-API-KEY"),
    resume_file: UploadFile = File(...),
    jd_file: UploadFile = File(None),
    jd_text: str = Form(None)
):
    verify_api_key(api_key)

    # --- Resume ---
    resume_bytes = await resume_file.read()
    if resume_file.filename.lower().endswith(".pdf"):
        resume_text = extract_text_from_pdf(io.BytesIO(resume_bytes))
    elif resume_file.filename.lower().endswith(".docx"):
        resume_text = extract_text_from_docx(io.BytesIO(resume_bytes))
    elif resume_file.filename.lower().endswith(".txt"):
        resume_text = resume_bytes.decode("utf-8")
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported resume file type"})

    # --- JD ---
    if jd_file:
        jd_bytes = await jd_file.read()
        if jd_file.filename.lower().endswith(".pdf"):
            jd_text_raw = extract_text_from_pdf(io.BytesIO(jd_bytes))
        elif jd_file.filename.lower().endswith(".docx"):
            jd_text_raw = extract_text_from_docx(io.BytesIO(jd_bytes))
        elif jd_file.filename.lower().endswith(".txt"):
            jd_text_raw = jd_bytes.decode("utf-8")
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported JD file type"})
    elif jd_text:
        jd_text_raw = jd_text
    else:
        return JSONResponse(status_code=400, content={"error": "Provide JD file or text"})

    return analyze_resume(resume_text, jd_text_raw)

# ---------- 2) JSON: resume_text + jd_text ----------
class ResumeRequest(BaseModel):
    resume_text: str
    jd_text: str

@app.post("/check_resume_json")
async def check_resume_json(
    payload: ResumeRequest,
    api_key: str = Header(..., alias="X-API-KEY")
):
    verify_api_key(api_key)
    return analyze_resume(payload.resume_text, payload.jd_text)

# ---------- Helpers for Streamlit ----------
def extract_keywords(jd_text: str):
    return extract_keywords_from_jd(clean_text(jd_text))

def calculate_match_score(resume_text: str, jd_text: str):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)
    jd_keywords = extract_keywords_from_jd(jd_clean)
    kw_score = keyword_match_score_loose(resume_clean, jd_keywords)
    tfidf = tfidf_score_boost(resume_clean, jd_clean)
    return round(final_score(kw_score, tfidf), 2)

def find_missing_keywords(resume_text: str, jd_keywords: list):
    resume_clean = clean_text(resume_text)
    return [s for s in jd_keywords if not re.search(r'\b' + re.escape(s) + r'\b', resume_clean)]

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("resume:app", host="127.0.0.1", port=8000, reload=True)
