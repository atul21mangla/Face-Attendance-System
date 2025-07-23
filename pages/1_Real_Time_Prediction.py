import streamlit as st
import cv2
import time
import threading
import numpy as np
import face_rec
from PIL import Image

st.set_page_config(page_title='Real-Time Predictions', layout='wide')

st.markdown("""
    <style>
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            color: #000000;
        }
        .stButton>button {
            background-color: #636efa;
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
        .status-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
    <div class="footer">🔒 Attendance System • OpenCV Camera</div>
""", unsafe_allow_html=True)

st.header("🎥 Real-Time Face Recognition System (OpenCV)")

# Initialize session state
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'redis_data_loaded' not in st.session_state:
    st.session_state.redis_data_loaded = False
if 'redis_face_db' not in st.session_state:
    st.session_state.redis_face_db = None

# Load Redis Data
st.subheader("📊 Database Connection")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔄 Load Redis Data"):
        with st.spinner('Retrieving Data from Redis DB...'):
            try:
                redis_face_db = face_rec.retrive_data(name='academy:register')
                st.session_state.redis_face_db = redis_face_db
                st.session_state.redis_data_loaded = True
                st.success("✅ Data successfully retrieved!")
            except Exception as e:
                st.error(f"❌ Failed to retrieve data: {str(e)}")
                st.session_state.redis_data_loaded = False

with col2:
    if st.session_state.redis_data_loaded:
        st.markdown('<div class="status-success">✅ Database loaded successfully</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-error">❌ Database not loaded</div>', unsafe_allow_html=True)

# Display registered data
if st.session_state.redis_data_loaded and st.session_state.redis_face_db is not None:
    with st.expander("🔍 View Registered Data"):
        st.dataframe(st.session_state.redis_face_db)

st.markdown("---")

# Camera Section
st.subheader("📹 Camera Controls")

if not st.session_state.redis_data_loaded:
    st.warning("⚠️ Please load the database first before starting the camera.")
else:
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🎥 Start Camera", disabled=st.session_state.camera_running):
            st.session_state.camera_running = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_running):
            st.session_state.camera_running = False
            st.rerun()
    
    with col3:
        camera_source = st.selectbox("📷 Camera Source", [0, 1, 2], index=0)

    # Camera settings
    st.subheader("⚙️ Settings")
    col1, col2 = st.columns([1, 1])
    with col1:
        waitTime = st.slider("Log Save Interval (seconds)", 1, 10, 3)
        recognition_threshold = st.slider("Recognition Threshold", 0.1, 1.0, 0.4, 0.1)
    with col2:
        frame_width = st.selectbox("Frame Width", [640, 1280, 1920], index=0)
        frame_height = st.selectbox("Frame Height", [480, 720, 1080], index=0)

    # Initialize face recognition
    if 'realtimepred' not in st.session_state:
        st.session_state.realtimepred = face_rec.RealTimePred()
    
    setTime = time.time()

    # Camera streaming
    if st.session_state.camera_running:
        st.subheader("📡 Live Camera Feed")
        
        # Create placeholders
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            st.error("❌ Cannot open camera. Please check if camera is available and not being used by another application.")
            st.session_state.camera_running = False
        else:
            status_placeholder.success("✅ Camera connected successfully!")
            
            # Camera loop
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Failed to capture frame from camera")
                    break
                
                # Process frame for face recognition
                try:
                    pred_img = st.session_state.realtimepred.face_prediction(
                        frame, 
                        st.session_state.redis_face_db,
                        'facial_features', 
                        ['Name', 'Role'], 
                        thresh=recognition_threshold
                    )
                    
                    # Save logs periodically
                    current_time = time.time()
                    if current_time - setTime >= waitTime:
                        try:
                            st.session_state.realtimepred.saveLogs_redis()
                            setTime = current_time
                            print("Saved logs to Redis DB")
                        except Exception as e:
                            print(f"Error saving logs: {e}")
                    
                    # Convert BGR to RGB for Streamlit
                    pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    frame_placeholder.image(pred_img_rgb, channels="RGB", use_container_width=True)
                    
                    # Calculate and display FPS
                    frame_count += 1
                    if frame_count % 30 == 0:  # Update every 30 frames
                        elapsed_time = current_time - start_time
                        fps = frame_count / elapsed_time
                        status_placeholder.success(f"✅ Camera running | FPS: {fps:.1f} | Frames: {frame_count}")
                
                except Exception as e:
                    st.error(f"❌ Error processing frame: {str(e)}")
                    break
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
            
            # Cleanup
            cap.release()
            frame_placeholder.empty()
            if st.session_state.camera_running:
                status_placeholder.info("📹 Camera stopped")
            st.session_state.camera_running = False

# Alternative: Image Upload for Testing
st.markdown("---")
st.subheader("📤 Test with Uploaded Images")

uploaded_files = st.file_uploader(
    "Upload test images", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files and st.session_state.redis_data_loaded:
    cols = st.columns(min(len(uploaded_files), 3))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx % 3]:
            # Convert uploaded file to opencv format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Process image
                pred_img = st.session_state.realtimepred.face_prediction(
                    img, 
                    st.session_state.redis_face_db,
                    'facial_features', 
                    ['Name', 'Role'], 
                    thresh=recognition_threshold
                )
                
                # Convert BGR to RGB for display
                pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
                st.image(pred_img_rgb, caption=uploaded_file.name, use_container_width=True)
            else:
                st.error(f"❌ Could not process {uploaded_file.name}")

# System Information
st.markdown("---")
st.subheader("🖥️ System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Camera Status**")
    if st.session_state.camera_running:
        st.success("🟢 Running")
    else:
        st.error("🔴 Stopped")

with col2:
    st.info("**Database Status**")
    if st.session_state.redis_data_loaded:
        st.success(f"🟢 {len(st.session_state.redis_face_db)} records")
    else:
        st.error("🔴 Not loaded")

with col3:
    st.info("**Recognition Threshold**")
    st.write(f"📊 {recognition_threshold}")

# Troubleshooting
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    **If camera doesn't work:**
    1. **Check camera permissions** in your browser/OS
    2. **Close other applications** using the camera (Zoom, Teams, etc.)
    3. **Try different camera sources** (0, 1, 2)
    4. **Restart Streamlit** if camera gets stuck
    
    **If face recognition doesn't work:**
    1. **Lower the recognition threshold** (try 0.3 or 0.2)
    2. **Check lighting conditions** (good lighting improves accuracy)
    3. **Verify database data** using the expand option above
    4. **Test with uploaded images** first
    
    **Performance tips:**
    1. **Lower resolution** for better performance
    2. **Increase log interval** to reduce Redis writes
    3. **Ensure stable lighting** for consistent results
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>💡 <strong>Note:</strong> This version uses OpenCV for camera access instead of WebRTC</p>
    <p>🔒 All processing is done locally - your data remains secure</p>
</div>
""", unsafe_allow_html=True)