# option/login.py
import streamlit as st
import pymysql
import hashlib

# ---- Database Connection ----
connection = pymysql.connect(host='localhost', user='root', password='1234', db='cv')
cursor = connection.cursor()

# ---- Create users table if not exists ----
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);
""")

# ---- Password Hashing ----
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hashlib.sha256(password.encode()).hexdigest() == hashed

# ---- Register ----
def register_user(username: str, email: str, password: str):
    try:
        hashed = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, email, hashed)
        )
        connection.commit()
        return True, None
    except pymysql.err.IntegrityError:
        return False, "Username or email already exists"
    except Exception as e:
        return False, str(e)

# ---- Authenticate ----
def get_user_by_identifier(identifier: str):
    cursor.execute(
        "SELECT id, username, email, password FROM users WHERE username=%s OR email=%s",
        (identifier, identifier)
    )
    return cursor.fetchone()

def authenticate(identifier: str, password: str):
    row = get_user_by_identifier(identifier)
    if row:
        user_id, uname, uemail, hashed = row
        if verify_password(password, hashed):
            return {"id": user_id, "username": uname, "email": uemail}
    return None

# ---- Login/Register UI ----
def login_register_ui():
    st.subheader("🔐 Login / Register")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user" not in st.session_state:
        st.session_state.user = None

    # If already logged in -> show welcome + logout (hide the forms)
    if st.session_state.logged_in and st.session_state.user:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success(f"Welcome, {st.session_state.user['username']}!")
        with col2:
            # unique key so button state doesn't clash with any other button
            if st.button("🚪 Logout", key="logout_top"):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.rerun()
        return

    # Not logged in: show login/register UI
    choice = st.radio("Choose action:", ["Login", "Register"], index=0, horizontal=True)

    if choice == "Register":
        with st.form("register_form", clear_on_submit=False):
            reg_username = st.text_input("Choose a username")
            reg_email = st.text_input("Email")
            reg_password = st.text_input("Password", type="password")
            reg_confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account")

        if submitted:
            if not reg_username or not reg_email or not reg_password:
                st.error("Please fill all fields.")
            elif reg_password != reg_confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(reg_username.strip(), reg_email.strip(), reg_password)
                if ok:
                    st.success("Account created! You can now log in.")
                else:
                    st.error(f"Registration failed: {msg}")

    else:  # Login
        with st.form("login_form", clear_on_submit=False):
            login_id = st.text_input("Username or Email")
            login_pw = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            if not login_id or not login_pw:
                st.error("Please enter credentials.")
            else:
                user = authenticate(login_id.strip(), login_pw)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.success(f"Welcome, {user['username']}!")
                    # optionally reredirect page or rerun to apply changes immediately
                    st.rerun()
                else:
                    st.error("Invalid username/email or password.")
