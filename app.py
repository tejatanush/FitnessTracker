import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import threading
import subprocess
import simpleaudio as sa
from PIL import Image
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer, util
import mediapipe as mp
from Pushups.pushups_helper import calculate_angle,PushupAnalyzer,PiperTTS,play_audio_async,draw_pose,display_info,check_relevancy,bot,run_chatbot,display_exercise_gif
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Fitness AI Trainer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader>div>div>div>button {
        background-color: #2196F3;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    .video-container {
        position: relative;
        padding-bottom: 56.25%;
        height: 0;
        overflow: hidden;
    }
    .video-container video {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }
    .feedback-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    /* Style for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
        background-color: #f0f2f6;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    /* Style for exercise GIF */
    .exercise-gif {
        max-width: 100%;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = ""
if 'pushup_count' not in st.session_state:
    st.session_state.pushup_count = 0

# App title
st.title("🏋️ Fitness AI Trainer")
st.markdown("""
    **Real-time exercise analysis** with pose detection and AI-powered feedback.
    """)

# Create tabs for different exercises
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Pushups", 
    "Squats", 
    "Deadlift", 
    "Skipping", 
    "Plank", 
    "Lunges", 
    "High Knees"
])

# Pushups Tab
with tab1:
    # Display pushups GIF
    display_exercise_gif("Pushups")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        analysis_mode = st.radio(
            "Select Mode",
            ["Video Analysis", "Live Camera", "Chat with AI Trainer"],
            index=0
        )
        
        if analysis_mode != "Chat with AI Trainer":
            st.subheader("Pushup Settings")
            lockout_thresh = st.slider("Lockout Threshold (degrees)", 150, 180, 160)
            bottom_thresh = st.slider("Bottom Threshold (degrees)", 70, 110, 90)
            
            st.subheader("Audio Feedback")
            audio_enabled = st.checkbox("Enable voice feedback", True)
            if audio_enabled:
                tts = PiperTTS()

    # Main content for Pushups tab
    if analysis_mode == "Video Analysis":
        uploaded_file = st.file_uploader("Upload your workout video", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Save uploaded file to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                temp_file_path=tfile.name
            try:
                analyzer = PushupAnalyzer()
                analyzer.lockout_threshold = lockout_thresh
                analyzer.bottom_threshold = bottom_thresh
                feedback_display_frames = 0
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                feedback_container = st.empty()
                
                feedbacks = set()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    if results.pose_landmarks:
                        draw_pose(frame, results.pose_landmarks)
                        landmarks = results.pose_landmarks.landmark
                        h, w, _ = frame.shape

                        try:
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
                            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h]
                            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h]
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h]

                            completed, feedback, elbow_angle, torso_angle = analyzer.process_frame(shoulder, elbow, wrist, hip)

                            if completed:
                                st.session_state.pushup_count = analyzer.pushup_count
                                feedbacks.add(feedback)
                                feedback_display_frames=30
                                if audio_enabled:
                                    tts.speak(feedback)
                                    threading.Thread(target=play_audio_async).start()

                            display_info(frame, analyzer.pushup_count, feedback if feedback_display_frames>0 else "", elbow_angle, torso_angle)
                            if feedback_display_frames > 0:
                                feedback_display_frames -= 1
                        
                        except (IndexError, AttributeError):
                            pass

                    stframe.image(frame, channels="BGR", use_container_width=True)
                
                cap.release()
                print(feedbacks)
                if feedbacks:
                    with st.spinner("Generating detailed analysis..."):
                        analysis = bot(feedbacks, exercise="pushups")
                        st.session_state.analysis_results = "\n\n".join(analysis)
                        
                    st.subheader("Detailed Analysis")
                    for i, response in enumerate(analysis):
                        with st.expander(f"Feedback Analysis {i+1}"):
                            st.markdown(response)
                            
                    st.session_state.feedback_history = list(feedbacks)
                else:
                    st.warning("No feedback was generated from the video.")
            finally:
                if os.path.exists(temp_file_path):
                    for _ in range(3):
                        try:
                            os.unlink(temp_file_path)
                            break
                        except PermissionError:
                            time.sleep(0.1)
    elif analysis_mode == "Live Camera":
        st.warning("Live camera feature requires running the app locally")
        run_camera = st.checkbox("Start Camera")
        
        if run_camera:
            analyzer = PushupAnalyzer()
            analyzer.lockout_threshold = lockout_thresh
            analyzer.bottom_threshold = bottom_thresh
            
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            feedback_container = st.empty()
            
            feedbacks = set()
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    draw_pose(frame, results.pose_landmarks)
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = frame.shape

                    try:
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h]
                        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h]

                        completed, feedback, elbow_angle, torso_angle = analyzer.process_frame(shoulder, elbow, wrist, hip)

                        if completed:
                            st.session_state.pushup_count = analyzer.pushup_count
                            feedbacks.add(feedback)
                            if audio_enabled:
                                tts.speak(feedback)
                                threading.Thread(target=play_audio_async).start()

                        display_info(frame, analyzer.pushup_count, feedback, elbow_angle, torso_angle)
                        
                    except (IndexError, AttributeError):
                        pass

                stframe.image(frame, channels="BGR", use_column_width=True)
                
            cap.release()

    elif analysis_mode == "Chat with AI Trainer":
        st.subheader("Chat with Your AI Fitness Trainer")
        
        if not st.session_state.analysis_results:
            st.warning("Please analyze a workout first to enable chat")
        else:
            user_input = st.text_input("Ask questions about your workout analysis:")
            
            if user_input:
                relevancy = check_relevancy(st.session_state.analysis_results, user_input)
                if relevancy != "Relevant":
                    st.warning("Please ask questions relevant to your workout analysis")
                else:
                    with st.spinner("Generating response..."):
                        response = run_chatbot(user_input, st.session_state.analysis_results)
                        st.session_state.chat_history.append(("You", user_input))
                        st.session_state.chat_history.append(("AI Trainer", response))
            
            if st.session_state.chat_history:
                st.subheader("Conversation History")
                for speaker, message in st.session_state.chat_history:
                    if speaker == "You":
                        st.markdown(f"**You:** {message}")
                    else:
                        st.markdown(f"**AI Trainer:** {message}")
                        st.markdown("---")

# Other exercise tabs (under development)
with tab2:
    display_exercise_gif("Squats")
    st.subheader("Squats Analysis")
    st.info("This feature is under development and will be available soon!")
    st.markdown("""
    **Proper Squat Form Tips:**
    - Keep your feet shoulder-width apart
    - Maintain a straight back throughout the movement
    - Lower until your thighs are parallel to the ground
    - Keep your knees aligned with your toes
    """)

with tab3:
    display_exercise_gif("Deadlift")
    st.subheader("Deadlift Analysis")
    st.info("This feature is under development and will be available soon!")
    st.markdown("""
    **Proper Deadlift Form Tips:**
    - Keep the bar close to your body
    - Maintain a neutral spine throughout
    - Drive through your heels
    - Engage your core and lats
    """)

with tab4:
    display_exercise_gif("Skipping")
    st.subheader("Skipping Analysis")
    st.info("This feature is under development and will be available soon!")
    st.markdown("""
    **Proper Skipping Form Tips:**
    - Keep your elbows close to your body
    - Use your wrists to turn the rope, not your arms
    - Jump just high enough to clear the rope
    - Maintain a steady rhythm
    """)

with tab5:
    display_exercise_gif("Plank")
    st.subheader("Plank Analysis")
    st.info("This feature is under development and will be available soon!")
    st.markdown("""
    **Proper Plank Form Tips:**
    - Keep your body in a straight line
    - Engage your core and glutes
    - Don't let your hips sag or rise too high
    - Keep your neck neutral
    """)

with tab6:
    display_exercise_gif("Lunges")
    st.subheader("Lunges Analysis")
    st.info("This feature is under development and will be available soon!")
    st.markdown("""
    **Proper Lunge Form Tips:**
    - Keep your upper body straight
    - Step far enough so your knee doesn't go past your toes
    - Lower until both knees are at 90 degrees
    - Push back up through your front heel
    """)

with tab7:
    display_exercise_gif("High Knees")
    st.subheader("High Knees Analysis")
    st.info("This feature is under development and will be available soon!")
    st.markdown("""
    **Proper High Knees Form Tips:**
    - Bring your knees up to hip level
    - Land softly on the balls of your feet
    - Keep your core engaged
    - Maintain a quick, steady pace
    """)

# Display pushup count in sidebar
with st.sidebar:
    st.markdown("---")
    st.metric(label="Total Pushups", value=st.session_state.pushup_count)
    if st.session_state.feedback_history:
        with st.expander("Recent Feedback"):
            for fb in st.session_state.feedback_history:
                st.text(fb)