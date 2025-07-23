import streamlit as st
import time

st.set_page_config(page_title='Attendance System', layout='wide')

st.markdown("""
    <style>
        
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            color: #000000;
        }
        .stButton>button {
            background-color: #2ECC71;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 16px;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #111;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
    </style>
    <div class="footer">🔒 Attendance System •</div>
""", unsafe_allow_html=True)

st.header('🧠 Attendance System using Face Recognition')

with st.spinner("Loading Models and Connecting to Redis DB..."):
    import face_rec
    time.sleep(2)

st.success('✅ Model loaded successfully!')
st.success('✅ Redis DB successfully connected!')
