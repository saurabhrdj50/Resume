import os
import re
import time
import requests
import streamlit as st
from datetime import datetime
from fpdf import FPDF  # fpdf2 version supports UTF-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional parser
try:
    from pyresparser import ResumeParser
    HAS_PYRESPARSER = True
except Exception:
    HAS_PYRESPARSER = False

# NLTK stopwords
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY", None)

# -------------------------
# AI text generation helpers
# -------------------------
def generate_with_openai(prompt, max_tokens=200, temperature=0.7):
    try:
        import openai
        if not OPENAI_KEY:
            return None
        openai.api_key = OPENAI_KEY
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

def generate_with_google_generative(prompt, model="text-bison-001", max_output_tokens=256, temperature=0.7):
    if not GOOGLE_KEY:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateText?key={GOOGLE_KEY}"
        payload = {
            "prompt": {"text": prompt},
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens
        }
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0].get("content", "").strip()
        return None
    except Exception:
        return None

def generate_text(prompt, prefer="openai"):
    if prefer == "openai":
        funcs = [generate_with_openai, generate_with_google_generative]
    else:
        funcs = [generate_with_google_generative, generate_with_openai]
    for fn in funcs:
        text = fn(prompt)
        if text:
            return text
    return "Accomplished professional with relevant experience and a strong track record."

# -------------------------
# ATS scoring
# -------------------------
def extract_keywords_simple(text, top_n=15):
    if not text or len(text) < 20:
        return []
    try:
        vec = TfidfVectorizer(max_df=0.85, stop_words="english", ngram_range=(1, 2), max_features=200)
        tfidf = vec.fit_transform([text])
        scores = dict(zip(vec.get_feature_names_out(), tfidf.toarray()[0]))
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [t[0] for t in top]
    except Exception:
        return []

def ats_score(resume_text, jd_text):
    if not resume_text or not jd_text:
        return 0.0
    try:
        cv = TfidfVectorizer(stop_words="english").fit_transform([resume_text, jd_text])
        sim = cosine_similarity(cv[0], cv[1])[0][0]
        return round(sim * 100, 2)
    except Exception:
        return 0.0

# -------------------------
# PDF Builder (UTF-8 Safe)
# -------------------------
class ResumePDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d')}", 0, 0, "C")

def build_pdf(payload: dict):
    pdf = ResumePDF()
    pdf.add_page()

    # Add Unicode font (download DejaVuSans.ttf and keep in same folder)
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    if not os.path.exists(font_path):
        raise FileNotFoundError("DejaVuSans.ttf not found in resume_builder.py folder.")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 12)

    # Name
    pdf.set_font("DejaVu", "B", 18)
    pdf.cell(0, 10, payload.get("name", ""), ln=True)
    pdf.set_font("DejaVu", "", 11)
    pdf.cell(0, 6, payload.get("contact_line", ""), ln=True)
    pdf.ln(4)

    # Summary
    summary = payload.get("summary", "")
    if summary:
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 7, "Summary", ln=True)
        pdf.set_font("DejaVu", "", 11)
        pdf.multi_cell(0, 6, summary)
        pdf.ln(2)

    # Skills
    skills = payload.get("skills", [])
    if skills:
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 7, "Skills", ln=True)
        pdf.set_font("DejaVu", "", 11)
        pdf.multi_cell(0, 6, " • ".join(skills))
        pdf.ln(2)

    # Experience
    exps = payload.get("experience", [])
    if exps:
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 7, "Experience", ln=True)
        pdf.set_font("DejaVu", "", 11)
        for e in exps:
            pdf.multi_cell(0, 6, f"• {e}")
            pdf.ln(1)
        pdf.ln(2)

    # Education
    ed = payload.get("education", [])
    if ed:
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 7, "Education", ln=True)
        pdf.set_font("DejaVu", "", 11)
        for e in ed:
            pdf.multi_cell(0, 6, f"• {e}")
            pdf.ln(1)

    return pdf.output(dest="S").encode("utf-8")

# -------------------------
# Streamlit UI
# -------------------------
def run_resume_builder():
    st.title("🛠 AI-powered Resume Builder")
    st.write("Upload an existing resume or fill the form; optionally paste a Job Description to tailor your resume.")

    name = st.text_input("Full name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    location = st.text_input("Location")
    summary = st.text_area("Summary")
    skills_text = st.text_input("Skills (comma separated)")
    experience_text = st.text_area("Experience (one per line)")
    education_text = st.text_area("Education (one per line)")
    jd_text = st.text_area("Job Description (optional)")

    if st.button("Generate Resume"):
        skills = [s.strip() for s in skills_text.split(",") if s.strip()]
        experience = [s.strip() for s in experience_text.splitlines() if s.strip()]
        education = [s.strip() for s in education_text.splitlines() if s.strip()]

        ats = ats_score(" ".join([summary, " ".join(skills), " ".join(experience), " ".join(education)]), jd_text) if jd_text else None

        if ats is not None:
            st.info(f"📊 ATS Match Score: {ats}%")

        pdf_payload = {
            "name": name,
            "contact_line": f"{email} | {phone} | {location}",
            "summary": summary,
            "skills": skills,
            "experience": experience,
            "education": education
        }
        pdf_bytes = build_pdf(pdf_payload)
        st.download_button("📥 Download Generated Resume (PDF)", data=pdf_bytes, file_name="resume.pdf", mime="application/pdf")
