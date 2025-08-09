import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------
# Admin Login Function
# ----------------------------
def admin_login():
    st.markdown("<h2 style='text-align:center;'>🔑 Admin Login</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", placeholder="Enter admin username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "1234":  # Change this for real security
                st.session_state.admin_logged_in = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")


# ----------------------------
# User Data Viewer
# ----------------------------
def view_saved_user_data(cursor):
    st.markdown("### 📋 Saved User Data")

    try:
        cursor.execute("SELECT * FROM user_data")
        rows = cursor.fetchall()

        if not rows:
            st.warning("No records found.")
            return

        # Get column names
        col_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=col_names)

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            email_filter = st.text_input("Search by Email")
        with col2:
            min_score = st.number_input("Min Score", min_value=0, max_value=100, value=0)
        with col3:
            max_score = st.number_input("Max Score", min_value=0, max_value=100, value=100)

        # Apply filters
        if email_filter and "Email_ID" in df.columns:
            df = df[df['Email_ID'].str.contains(email_filter, case=False, na=False)]
        if "resume_score" in df.columns:
            df = df[
                (pd.to_numeric(df['resume_score'], errors='coerce') >= min_score) &
                (pd.to_numeric(df['resume_score'], errors='coerce') <= max_score)
            ]

        # Display table
        st.dataframe(df, use_container_width=True)

        # CSV Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="user_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error fetching data: {e}")


# ----------------------------
# Dashboard Analytics
# ----------------------------
def dashboard_analytics(cursor):
    st.markdown("### 📊 Dashboard Overview")

    try:
        cursor.execute("SELECT resume_score, Predicted_Field, User_level, country FROM user_data")
        rows = cursor.fetchall()
        if not rows:
            st.warning("No data for analytics.")
            return

        # Decode BLOBs into strings
        decoded_rows = []
        for r in rows:
            score = r[0]
            predicted = r[1].decode('utf-8') if isinstance(r[1], bytes) else r[1]
            cand_level = r[2].decode('utf-8') if isinstance(r[2], bytes) else r[2]
            country = r[3]
            decoded_rows.append((score, predicted, cand_level, country))

        df = pd.DataFrame(decoded_rows, columns=["score", "predicted", "cand_level", "country"])

        # Basic metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Average Score", f"{pd.to_numeric(df['score'], errors='coerce').mean():.2f}")
        col3.metric("Unique Countries", df['country'].nunique())

        # Charts in tabs
        chart_tabs = st.tabs(["📌 Candidate Level", "🌎 Country", "🎯 Predicted Role"])

        with chart_tabs[0]:
            fig1 = px.pie(df, names="cand_level", title="Candidate Level Distribution")
            st.plotly_chart(fig1, use_container_width=True)

        with chart_tabs[1]:
            fig2 = px.pie(df, names="country", title="Country Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        with chart_tabs[2]:
            fig3 = px.pie(df, names="predicted", title="Predicted Role Distribution")
            st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading analytics: {e}")


# ----------------------------
# Main Admin Section
# ----------------------------
def run_admin_section(cursor, connection=None, get_csv_download_link=None):
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        admin_login()
        return

    # Admin Panel Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📋 Saved User Data", "🚪 Logout"])

    with tab1:
        dashboard_analytics(cursor)

    with tab2:
        view_saved_user_data(cursor)

    with tab3:
        if st.button("Logout", type="primary"):
            st.session_state.admin_logged_in = False
            st.rerun()
