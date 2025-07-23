import streamlit as st
import face_rec

st.set_page_config(page_title='Reporting', layout='wide')

st.markdown("""
    <style>
        
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            color: #000000;
        }
        .stButton>button {
            background-color: #F39C12;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
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

st.header("📊 Attendance Reports")

name = 'attendance:logs'
def load_logs(name, end=-1):
    logs_list = face_rec.r.lrange(name, start=0, end=end)
    return logs_list

tab1, tab2 = st.tabs(["👥 Registered Users", "📝 Attendance Logs"])

with tab1:
    if st.button('🔄 Refresh Registered Data'):
        with st.spinner("Loading from Redis..."):
            redis_face_db = face_rec.retrive_data(name='academy:register')
            st.success("✅ Data loaded successfully.")
            st.dataframe(redis_face_db[['Name', 'Role']])

with tab2:
    if st.button('🔄 Refresh Attendance Logs'):
        st.success("✅ Attendance Logs:")
        logs = load_logs(name=name)
        for i, entry in enumerate(logs[:50], 1):
            st.markdown(f"**{i}.** {entry.decode()}")
