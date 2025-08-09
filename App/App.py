# Developed by dnoobnerd [https://dnoobnerd.netlify.app] | Updated by Saurabh

import streamlit as st
import pandas as pd
import base64
import random
import time, datetime
import pymysql
import os
import io
from PIL import Image
from chatbot import chat_with_resume_context, general_chatbot
from option.user import run_user_section
from option.login import login_register_ui
from option.admin import run_admin_section
from option.resume_builder import run_resume_builder
from Courses import ds_course, web_course, android_course, ios_course, uiux_course
import plotly.express as px
import nltk

nltk.download('stopwords')

# Access secrets
db_host = st.secrets["DB_HOST"]
db_user = st.secrets["DB_USER"]
db_pass = st.secrets["DB_PASSWORD"]
gemini_key = st.secrets["GEMINI_API_KEY"]

st.write("Database host is:", db_host)


# -------------------- Styling --------------------
def local_css():
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #f8fbff 0%, #e6f0ff 100%);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
            color: white;
        }
        section[data-testid="stSidebar"] img {
            border-radius: 50%;
            margin-bottom: 15px;
            box-shadow: 0px 0px 10px rgba(255,255,255,0.4);
        }
        .app-header {
            font-size: 32px;
            font-weight: bold;
            color: white;
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            background: linear-gradient(90deg, #1e3c72, #2a5298);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            margin-bottom: 25px;
        }
        .hero-title {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .hero-subtitle {
            font-size: 16px;
            color: #555;
            margin-top: 0;
        }
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            color: white;
            margin-bottom: 10px;
        }
        div.stButton > button {
            background: linear-gradient(45deg, #6a11cb, #2575fc);
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            background: linear-gradient(45deg, #2575fc, #6a11cb);
            box-shadow: 0 0 10px rgba(37,117,252,0.6);
            transform: scale(1.05);
        }
        </style>
    """, unsafe_allow_html=True)

# -------------------- Utility Functions --------------------
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

# Database connection
connection = pymysql.connect(host='localhost', user='root', password='1234', db='cv')
cursor = connection.cursor()

# Insert resume data
def insert_data(sec_token, ip_add, host_name, dev_user, os_name_ver, latlong, city, state, country,
                act_name, act_mail, act_mob, name, email, res_score, timestamp, no_of_pages,
                reco_field, cand_level, skills, recommended_skills, courses, pdf_name):
    DB_table_name = 'user_data'
    insert_sql = f"INSERT INTO {DB_table_name} VALUES (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    rec_values = (str(sec_token), str(ip_add), host_name, dev_user, os_name_ver, str(latlong), city, state, country,
                  act_name, act_mail, act_mob, name, email, str(res_score), timestamp, str(no_of_pages),
                  reco_field, cand_level, skills, recommended_skills, courses, pdf_name)
    cursor.execute(insert_sql, rec_values)
    connection.commit()

# Insert feedback
def insertf_data(feed_name, feed_email, feed_score, comments, Timestamp):
    DBf_table_name = 'user_feedback'
    insertfeed_sql = f"INSERT INTO {DBf_table_name} VALUES (0,%s,%s,%s,%s,%s)"
    rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
    cursor.execute(insertfeed_sql, rec_values)
    connection.commit()

# -------------------- Main App --------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon='./Logo/recommend.png')
local_css()

def run():
    st.markdown('<div class="app-header">📄 AI Resume Analyzer – Smart Career Insights</div>', unsafe_allow_html=True)

    # Hero Section
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(Image.open('./Logo/RESUM.png'), use_container_width=True)
    with col2:
        st.markdown('<p class="hero-title">Empowering Careers with AI & ML</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Get insights, recommendations, and predictions for your career with AI-powered resume analysis.</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-title">📌 Choose Something...</div>', unsafe_allow_html=True)
    activities = ["User", "General Chatbot", "Resume Builder", "Feedback", "About", "Admin"]
    choice = st.sidebar.selectbox(
        "Choose an option:",
        activities,
        label_visibility="collapsed"
    )


    st.sidebar.markdown('<b>Built with 🤍 by <a href="https://saurabh.free.nf/?i=1" style="text-decoration: none; color: white;">Saurabh</a></b>', unsafe_allow_html=True)
    st.sidebar.markdown('<p>👥 Visitors</p><img src="https://counter9.stat.ovh/private/freecounterstat.php?c=t2xghr8ak6lfqt3kgru233378jya38dy" width="80px" />', unsafe_allow_html=True)

    # Navigation
    if choice == 'User':
        login_register_ui()
        if st.session_state.get("logged_in") and st.session_state.get("user"):
            st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.update({"logged_in": False, "user": None}))
            run_user_section(cursor, insert_data)
        else:
            st.info("Please register or login to access the Resume Analyzer.")

    elif choice == "General Chatbot":
        st.header("💬 General Chatbot")
        general_chatbot()

    elif choice == 'Feedback':
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        with st.form("feedback_form"):
            feed_name = st.text_input('Name')
            feed_email = st.text_input('Email')
            feed_score = st.slider('Rate Us From 1 - 5', 1, 5)
            comments = st.text_area('Comments')
            if st.form_submit_button("Submit"):
                insertf_data(feed_name, feed_email, feed_score, comments, timestamp)
                st.success("Thanks! Your Feedback was recorded.")
                st.balloons()

        query = 'SELECT * FROM user_feedback'
        plotfeed_data = pd.read_sql(query, connection)
        st.subheader("**Past User Ratings**")
        fig = px.pie(values=plotfeed_data.feed_score.value_counts(), names=plotfeed_data.feed_score.unique(),
                     title="User Rating Distribution", color_discrete_sequence=px.colors.sequential.Aggrnyl)
        st.plotly_chart(fig)
        st.subheader("**User Comments**")
        st.dataframe(plotfeed_data[['feed_name', 'comments']], width=1000)

    elif choice == 'About':
        st.subheader("**About The Tool - AI RESUME ANALYZER**")
        st.markdown("""
        This tool parses resume data using NLP and ML to:
        - Identify skills and missing skills
        - Recommend courses
        - Predict career level
        Built with ❤️ by Saurabh
        """)

    elif choice == 'Admin':
        run_admin_section(cursor, connection, get_csv_download_link)

    elif choice == "Resume Builder":
        run_resume_builder()

run()
