
# streamlit_app.py
import streamlit as st
from app.Resume import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_keywords,
    calculate_match_score,
    find_missing_keywords
)

st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="centered")
st.title("📄 AI Resume Job Match System (ATS Booster)")

resume_file = st.file_uploader("Upload Your Resume", type=["pdf", "docx", "txt"])
jd_text = st.text_area("Paste Job Description")

if resume_file and jd_text:
    if resume_file.name.lower().endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_file)
    elif resume_file.name.lower().endswith(".docx"):
        resume_text = extract_text_from_docx(resume_file)
    elif resume_file.name.lower().endswith(".txt"):
        resume_text = resume_file.read().decode("utf-8")
    else:
        st.error("Unsupported file format. Please upload PDF, DOCX, or TXT.")
        st.stop()

    jd_keywords = extract_keywords(jd_text)
    score = calculate_match_score(resume_text, jd_text)
    missing = find_missing_keywords(resume_text, jd_keywords)

    st.subheader(f"✅ Match Score: {score}%")
    st.write("📌 Missing Keywords:")
    st.write(", ".join(missing) if missing else "None 🎉")
