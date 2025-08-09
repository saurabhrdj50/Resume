import streamlit as st
import google.generativeai as genai
from datetime import datetime

# ----------------------------------------
# Configure Gemini API
# ----------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")


# ----------------------------------------
# Inject Premium UI CSS
# ----------------------------------------
def chatbot_css():
    st.markdown("""
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
            max-height: 500px;
            overflow-y: auto;
            padding: 20px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(10px);
            box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }

        .user-bubble {
            align-self: flex-end;
            background: linear-gradient(to right, #00c6ff, #0072ff);
            color: white;
            padding: 12px 18px;
            border-radius: 20px 20px 0 20px;
            max-width: 75%;
            font-size: 1rem;
            text-align: right;
            margin-left: auto;
            animation: fadeIn 0.3s ease-in-out;
        }

        .bot-bubble {
            align-self: flex-start;
            background: #ffffff;
            color: #222;
            padding: 12px 18px;
            border-radius: 20px 20px 20px 0;
            max-width: 75%;
            font-size: 1rem;
            margin-right: auto;
            border: 1px solid #ddd;
            animation: fadeIn 0.3s ease-in-out;
        }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }

        .stTextInput > div > div > input {
            border-radius: 25px;
            padding: 12px 20px;
            font-size: 1rem;
            border: 1px solid #ccc;
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(6px);
        }

        button[kind="primary"] {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-weight: bold;
            border-radius: 25px;
            padding: 8px 20px;
            transition: 0.3s ease;
            margin-top: 5px;
        }

        button[kind="primary"]:hover {
            transform: scale(1.05);
            box-shadow: 0px 4px 12px rgba(118, 75, 162, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)


# ----------------------------------------
# Generic Chat UI
# ----------------------------------------
def chat_ui(session_key, title, system_prompt=None):
    chatbot_css()
    st.markdown(f"## {title}")

    # Clear chat
    if st.button(f"🗑️ Clear {title}", use_container_width=True):
        st.session_state[session_key] = []
        st.rerun()

    if session_key not in st.session_state:
        st.session_state[session_key] = []

    # Display chat bubbles
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for entry in st.session_state[session_key]:
        if len(entry) == 3:
            sender, message, timestamp = entry
        else:
            sender, message = entry
            timestamp = ""

        bubble_class = "user-bubble" if sender == "You" else "bot-bubble"
        st.markdown(f'''
            <div class="{bubble_class}">
                <div>{message}</div>
                <div style="font-size: 0.75rem; color: gray; margin-top: 5px;">{timestamp}</div>
            </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input form
    with st.form(f"form_{session_key}", clear_on_submit=True):
        prompt = st.text_input("Type your message", key=f"input_{session_key}")
        submitted = st.form_submit_button("Send")

    if submitted and prompt:
        with st.spinner("Generating response..."):
            try:
                full_prompt = f"""
                {system_prompt if system_prompt else ""}

                User question:
                {prompt}
                """
                response = model.generate_content(full_prompt)
                reply = response.text.strip()

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state[session_key].append(("You", prompt, timestamp))
                st.session_state[session_key].append(("AI", reply, timestamp))
                st.rerun()

            except Exception as e:
                st.error(f"❌ Gemini API Error: {str(e)}")

    # Save chat history
    if st.session_state[session_key]:
        chat_text = ""
        for entry in st.session_state[session_key]:
            if len(entry) == 3:
                sender, message, timestamp = entry
                chat_text += f"[{timestamp}] {sender}: {message}\n"
            else:
                sender, message = entry
                chat_text += f"{sender}: {message}\n"

        st.download_button("💾 Download Chat", data=chat_text, file_name=f"{session_key}_chat.txt", mime="text/plain")


# ----------------------------------------
# Resume Chatbot Wrapper
# ----------------------------------------
def chat_with_resume_context(resume_text):
    system_prompt = f"""
    You are an AI assistant helping a candidate improve their resume.
    Here's the candidate's resume summary:
    {resume_text}
    """
    chat_ui("resume_chat", "🤖 Resume Chat Assistant", system_prompt)


# ----------------------------------------
# General-purpose Chatbot Wrapper
# ----------------------------------------
def general_chatbot():
    chat_ui("general_chat", "💬 General Chatbot")
