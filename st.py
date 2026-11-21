import streamlit as st
import cv2
import pandas as pd
import time
from deepface import DeepFace

# 1. Project Overview & UI Setupp
st.set_page_config(page_title="AI Emotion Analytics", layout="wide")
st.title("Real-Time AI Emotion Analytics Dashboard")
st.markdown("Monitoring live video using Deep Learning to detect and analyze human emotions.")

# Create columns: Left for Video, Right for the Analytics Chart
col1, col2 = st.columns([2, 1])
video_placeholder = col1.empty()
chart_placeholder = col2.empty()

# Load the lightweight face detector for real-time bounding boxes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sidebar controls
st.sidebar.header("Control Panel")
run_monitor = st.sidebar.toggle("Activate Camera")
process_every_n_frames = st.sidebar.slider("Analysis Frequency", 5, 30, 15, 
                                           help="Lower number = faster emotion updates, but uses more CPU.")

# 4. Workflow & Logic
if run_monitor:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    dominant_emotion = ""  # Stores the latest emotion text
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
            
        frame_count += 1
        
        # Convert to grayscale strictly for the Haar Cascade face detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw bounding boxes and text around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if dominant_emotion:
                # Overlay the AI's emotion guess right above the face box
                cv2.putText(frame, dominant_emotion.capitalize(), (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Deep Learning Engine: Run DeepFace analysis on a schedule to prevent lag
        if frame_count % process_every_n_frames == 0 and len(faces) > 0:
            try:
                # Analyze the original BGR frame (DeepFace handles its own preprocessing)
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                # DeepFace returns a list if it spots multiple faces; we'll grab the first one
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                emotions_dict = analysis['emotion']
                dominant_emotion = analysis['dominant_emotion']
                
                # Data Visualization: Convert emotion scores to a Pandas DataFrame
                df = pd.DataFrame(list(emotions_dict.items()), columns=['Emotion', 'Confidence Level'])
                df.set_index('Emotion', inplace=True)
                
                # Update the right column with the dynamic bar chart
                with chart_placeholder.container():
                    st.subheader("Live Confidence Levels")
                    st.bar_chart(df)
                    
            except Exception as e:
                # If a frame is too blurry for DeepFace, we safely pass and wait for the next one
                pass 

        # Render the final processed frame to the Streamlit UI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB")
        
        time.sleep(0.01)
        
    cap.release()
else:
    st.info("System Standby. Toggle 'Activate Camera' in the sidebar to begin analysis.")