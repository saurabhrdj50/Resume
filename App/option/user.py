# option/user.py
# Advanced resume analyzer with ML/NLP features (graceful fallbacks)
# Keep signature: run_user_section(cursor, insert_data)

import streamlit as st
import time, datetime, secrets, socket, os, platform, random, base64, io, math
from pathlib import Path

# PDF parsing
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter

# Optional parser (nice to have)
try:
    from pyresparser import ResumeParser
    HAS_PYRESPARSER = True
except Exception:
    HAS_PYRESPARSER = False

# NLP and ML libraries (optional/enhanced features)
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    HAS_NLTK = True
except Exception:
    STOPWORDS = set()
    HAS_NLTK = False

# TF-IDF / cosine fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# RAKE (optional)
try:
    from rake_nltk import Rake
    HAS_RAKE = True
except Exception:
    HAS_RAKE = False

# Sentence-transformers (optional, much better semantic similarity)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    EMBED_MODEL = None
    HAS_SENTENCE_TRANSFORMERS = False

# Other bits
from streamlit_tags import st_tags
from geopy.geocoders import Nominatim
import geocoder

# Courses and videos (reuse your existing Courses module if present)
try:
    from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
except Exception:
    # Minimal fallback lists if Courses.py isn't present
    ds_course = [("Intro to ML", "https://example.com/ml")]
    web_course = [("Web Dev Bootcamp", "https://example.com/web")]
    android_course = [("Android Basics", "https://example.com/android")]
    ios_course = [("iOS Basics", "https://example.com/ios")]
    uiux_course = [("UI/UX Design", "https://example.com/uiux")]
    resume_videos = ["https://www.youtube.com/watch?v=V1eYniJ0Rnk"]
    interview_videos = ["https://www.youtube.com/watch?v=R6cF7rI3e2k"]

# ---------- Utilities ----------
def pdf_reader(file_path: str) -> str:
    """Extract text from a PDF using pdfminer3. Returns plain text."""
    resource_manager = PDFResourceManager()
    fake_handle = io.StringIO()
    laparams = LAParams()
    converter = TextConverter(resource_manager, fake_handle, laparams=laparams)
    interpreter = PDFPageInterpreter(resource_manager, converter)
    try:
        with open(file_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                interpreter.process_page(page)
    except Exception:
        # fallback: return empty
        return ""
    text = fake_handle.getvalue()
    converter.close()
    fake_handle.close()
    return text or ""

def show_pdf(file_path: str):
    """Embed PDF in Streamlit."""
    try:
        with open(file_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        iframe = f"<iframe src='data:application/pdf;base64,{b64}' width='700' height='700' type='application/pdf'></iframe>"
        st.markdown(iframe, unsafe_allow_html=True)
    except Exception:
        st.error("Could not show PDF preview.")

# ---------- Keyword extraction ----------
def extract_keywords_rake(text: str, top_n=10):
    if not HAS_RAKE:
        return []
    r = Rake(stopwords=STOPWORDS) if HAS_NLTK else Rake()
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()[:top_n]
    return phrases

def extract_keywords_tfidf(text: str, n=10):
    """Extract candidate keywords using TF-IDF over sentences/phrases fallback."""
    if not text or len(text.split()) < 5:
        return []
    # split into simple "phrases" by punctuation for scoring
    # using naive approach: consider n-grams? simpler: top tfidf terms
    vec = TfidfVectorizer(max_df=0.85, stop_words='english', ngram_range=(1,2), max_features=200)
    tfidf = vec.fit_transform([text])
    scores = dict(zip(vec.get_feature_names_out(), tfidf.toarray()[0]))
    # sort by score
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [t[0] for t in top]

# ---------- Resume summarization (simple) ----------
def summarize_text(text: str, max_sentences=4):
    """If a transformer summarizer is available you'd plug it here.
    Fallback: pick first meaningful sentences."""
    if not text:
        return ""
    # naive sentence split
    sentences = [s.strip() for s in text.replace("\r", " ").split(".") if len(s.strip()) > 20]
    return ". ".join(sentences[:max_sentences]) + ("." if len(sentences) > 0 else "")

# ---------- Semantic similarity / role matching ----------
# Provide a small set of example job descriptions for matching (you can expand)
EXAMPLE_JOB_DESCRIPTIONS = {
    "Data Scientist": (
        "We are looking for a Data Scientist with experience in machine learning, "
        "statistical modeling, Python, pandas, scikit-learn, deep learning frameworks like TensorFlow or PyTorch."
    ),
    "Backend Web Developer": (
        "Looking for Backend Web Developer with Django or Node.js experience, REST APIs, databases (MySQL/Postgres), "
        "unit testing and deployment skills."
    ),
    "Frontend Web Developer": (
        "Frontend developer skilled in React, JavaScript, HTML/CSS, responsive design, and modern frontend tooling."
    ),
    "Mobile Developer (Android)": (
        "Android developer with Kotlin/Java experience, Android SDK, REST API integration and knowledge of app lifecycle."
    ),
    "UI/UX Designer": (
        "UI/UX Designer experienced with Figma/Adobe XD, prototyping, user research, wireframing and interaction design."
    ),
}

def embed_texts(texts):
    """Embed a list of texts. Use sentence-transformers if available, otherwise TF-IDF vectorizer."""
    if HAS_SENTENCE_TRANSFORMERS:
        emb = EMBED_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb
    else:
        # TF-IDF vectors (dense) as fallback
        vec = TfidfVectorizer(max_df=0.85, stop_words='english', ngram_range=(1,2))
        tfidf = vec.fit_transform(texts)
        return tfidf

def compute_similarity(resume_text: str, job_descs: dict, top_k=3):
    """Return a sorted list of (role, score) — higher score first."""
    if not resume_text or len(resume_text) < 20:
        return []
    roles = list(job_descs.keys())
    texts = [resume_text] + [job_descs[r] for r in roles]
    emb = embed_texts(texts)
    # similarity between first vector and others
    try:
        if HAS_SENTENCE_TRANSFORMERS:
            # emb is numpy array
            sims = (emb[0] @ emb[1:].T).tolist()  # cosine because normalized
        else:
            sims = cosine_similarity(emb[0], emb[1:])[0].tolist()
    except Exception:
        sims = [0.0] * len(roles)
    scored = list(zip(roles, sims))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    return scored_sorted

# ---------- Scoring & checklist ----------
SCORE_CHECKS = [
    (("objective", "summary"), 6, "Objective/Summary present"),
    (("education", "school", "college"), 12, "Education section"),
    (("experience", "work experience"), 16, "Experience section"),
    (("internship", "internships"), 6, "Internships"),
    (("skills", "skill"), 7, "Skills listed"),
    (("projects", "project"), 19, "Projects"),
    (("certification", "certifications"), 12, "Certifications"),
    (("achievements", "achievement"), 13, "Achievements"),
    (("hobbies",), 4, "Hobbies"),
    (("interest","interests"), 5, "Interests"),
]

def score_resume(resume_text: str):
    text = (resume_text or "").lower()
    score = 0
    check_results = []
    for keywords, points, label in SCORE_CHECKS:
        found = any(k in text for k in keywords)
        if found:
            score += points
        check_results.append((label, found, points if found else 0))
    score = min(100, score)
    return score, check_results

# ---------- Simple skill recommendations per role ----------
RECOMMENDATIONS = {
    "Data Scientist": ["Python", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "SQL", "Statistics"],
    "Backend Web Developer": ["Django/Flask", "REST APIs", "SQL", "Docker", "Unit Testing"],
    "Frontend Web Developer": ["React", "JavaScript (ES6+)", "HTML/CSS", "Webpack/Vite"],
    "Mobile Developer (Android)": ["Kotlin", "Android SDK", "REST APIs", "SQLite"],
    "UI/UX Designer": ["Figma", "Wireframing", "Prototyping", "User Research"],
}

COURSE_MAP = {
    "Data Scientist": ds_course,
    "Backend Web Developer": web_course,
    "Frontend Web Developer": web_course,
    "Mobile Developer (Android)": android_course,
    "UI/UX Designer": uiux_course,
}

# ---------- Main UI ----------
def run_user_section(cursor, insert_data):
    st.markdown("<h2 style='color:#021659;'>📥 Upload your resume — advanced analysis & AI suggestions</h2>", unsafe_allow_html=True)

    # Logout
    if "logged_in" in st.session_state and st.session_state.logged_in:
        col1, col2 = st.columns([6, 1])
        with col1:
            st.write(f"Logged in as: **{st.session_state.user.get('username', 'user')}**")
        with col2:
            if st.button("🚪 Logout", key="logout_user"):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.rerun()

    # Basic user inputs
    c1, c2 = st.columns(2)
    with c1:
        act_name = st.text_input("Your name*", value=(st.session_state.user.get('username') if st.session_state.get('user') else ""))
        act_mail = st.text_input("Your email*", value=(st.session_state.user.get('email') if st.session_state.get('user') else ""))
    with c2:
        act_mob = st.text_input("Mobile number*")

    # Environment info (best-effort)
    sec_token = secrets.token_urlsafe(12)
    try:
        host_name = socket.gethostname()
    except Exception:
        host_name = "unknown"
    try:
        ip_add = socket.gethostbyname(host_name)
    except Exception:
        ip_add = "127.0.0.1"
    try:
        dev_user = os.getlogin()
    except Exception:
        dev_user = ""
    os_name_ver = f"{platform.system()} {platform.release()}"

    # Geo best-effort
    latlong = None
    city = state = country = ""
    try:
        g = geocoder.ip('me')
        latlong = g.latlng
        if latlong:
            geolocator = Nominatim(user_agent="resume-analyzer")
            location = geolocator.reverse(latlong, language='en')
            if location and 'address' in location.raw:
                addr = location.raw['address']
                city = addr.get('city') or addr.get('town') or addr.get('village') or ""
                state = addr.get('state', "")
                country = addr.get('country', "")
    except Exception:
        pass

    st.markdown("---")
    uploaded = st.file_uploader("Choose your resume (PDF)", type=["pdf"])
    if not uploaded:
        st.info("Please upload a PDF resume to enable analysis.")
        return

    # Save upload
    uploads_dir = Path("./Uploaded_Resumes")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / uploaded.name
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # preview
    show_pdf(str(file_path))

    # parse with pyresparser if available (optional)
    parsed = {}
    with st.spinner("Parsing resume (this can take a few seconds)..."):
        time.sleep(0.8)
        if HAS_PYRESPARSER:
            try:
                parsed = ResumeParser(str(file_path)).get_extracted_data() or {}
            except Exception:
                parsed = {}
        # fallback: at least read full text
        resume_text = pdf_reader(str(file_path))
        if not parsed:
            parsed = {"name": "", "email": "", "mobile_number": "", "skills": []}
            # attempt naive email/mobile extraction?
            # (keeping minimal to avoid overcomplication)
    # display parsed basics
    st.header("🧾 Parsed Summary")
    st.write(f"**Name (detected):** {parsed.get('name','-')}")
    st.write(f"**Email (detected):** {parsed.get('email','-')}")
    st.write(f"**Mobile (detected):** {parsed.get('mobile_number','-')}")
    st.write(f"**Pages (detected):** {parsed.get('no_of_pages', 'unknown')}")
    st.markdown("----")

    # Keywords extraction
    st.subheader("🔎 Extracted Keywords & Keyphrases")
    keywords = []
    # RAKE first (if available)
    if HAS_RAKE:
        try:
            kw_rake = extract_keywords_rake(resume_text, top_n=15)
            keywords.extend(kw_rake)
        except Exception:
            pass
    # TF-IDF fallback
    try:
        kw_tfidf = extract_keywords_tfidf(resume_text, n=15)
        keywords.extend([k for k in kw_tfidf if k not in keywords])
    except Exception:
        pass
    # show tags
    if keywords:
        st_tags(label="Keywords detected", text="Auto-extracted keywords from resume", value=keywords, key="parsed_kw")
    else:
        st.write("No keywords could be extracted.")

    # Summarize resume
    st.subheader("✍️ Resume Summary (quick)")
    summary = summarize_text(resume_text, max_sentences=4)
    st.write(summary or "No summary available (resume text too short or not extractable).")

    # Skill detection via parsed skills (from pyresparser) or keywords
    st.subheader("🛠 Skills & Recommendations")
    parsed_skills = parsed.get("skills") or []
    st_tags(label="Detected skills", text="Skills identified in your resume", value=parsed_skills or keywords[:10], key="det_skills")

    # Role matching (semantic similarity)
    st.subheader("🎯 Role suggestions (based on resume content)")
    sim_results = compute_similarity(resume_text, EXAMPLE_JOB_DESCRIPTIONS, top_k=3)
    if sim_results:
        for role, score in sim_results:
            pct = f"{score*100:.1f}%" if score <= 1 else f"{score:.2f}"
            st.markdown(f"**{role}** — match score: **{pct}**")
            # recommended skills and courses
            rec_sk = RECOMMENDATIONS.get(role, [])
            if rec_sk:
                st.markdown("Recommended skills:")
                st.write(", ".join(rec_sk))
            # show a curated course list (if available)
            if role in COURSE_MAP:
                try:
                    st.markdown("Recommended courses:")
                    for name, link in COURSE_MAP[role]:
                        st.markdown(f"- [{name}]({link})")
                except Exception:
                    pass
    else:
        st.write("No role suggestions (resume may be too short).")

    # Resume scoring & checklist
    st.subheader("📊 Resume Score & Checklist")
    score, checks = score_resume(resume_text)
    # Show checklist
    for label, ok, pts in checks:
        status = "✅" if ok else "⚠️"
        st.markdown(f"{status} **{label}** {'(' + str(pts) + ' pts)' if ok else ''}")
    # Animated progress bar (short)
    bar = st.progress(0)
    for i in range(int(score)+1):
        bar.progress(i)
        time.sleep(0.003)
    st.success(f"Your Resume Score: **{score}/100**")
    st.info("Score is a heuristic based on presence of standard resume sections. Improve sections with ⚠️ to raise score.")

    # Actionable suggestions (basic)
    st.subheader("🛠 Actionable Suggestions")
    missing = [label for label, ok, _ in checks if not ok]
    if missing:
        st.markdown("These sections are missing or unclear — consider adding/editing them:")
        for m in missing:
            st.write(f"- {m}")
    else:
        st.write("All basic sections detected. Good job!")

    # Save to DB via provided insert_data function
    st.markdown("---")
    if st.button("✅ Save Analysis & Record to DB"):
        ts = time.time()
        cur_date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        cur_time = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        timestamp = f"{cur_date}_{cur_time}"
        try:
            insert_data(
                str(sec_token),
                str(ip_add),
                host_name,
                dev_user,
                os_name_ver,
                str(latlong),
                city,
                state,
                country,
                act_name,
                act_mail,
                act_mob,
                parsed.get("name", ""),
                parsed.get("email", ""),
                str(score),
                timestamp,
                str(parsed.get("no_of_pages", 1)),
                # predicted field: top role if any
                (sim_results[0][0] if sim_results else "NA"),
                # candidate level heuristic: derived from resume text
                ("Experienced" if "experience" in (resume_text or "").lower() else ("Intermediate" if "intern" in (resume_text or "").lower() else "Fresher")),
                str(parsed_skills),
                str(list(set(keywords) - set(parsed_skills))),
                str([r for r,_ in sim_results]),
                uploaded.name,
            )
            st.success("Saved to DB successfully.")
        except Exception as e:
            st.error(f"Could not save to DB: {e}")

    # Bonus videos & chatbot
    with st.expander("🎥 Resume & Interview Tips"):
        try:
            st.video(random.choice(resume_videos))
            st.video(random.choice(interview_videos))
        except Exception:
            pass

    # Hand off to chatbot if available
    try:
        from chatbot import chat_with_resume_context
        chat_with_resume_context(resume_text)
    except Exception:
        st.info("AI chat with resume context not available right now.")
