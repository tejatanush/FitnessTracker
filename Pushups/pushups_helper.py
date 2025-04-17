import numpy as np
import streamlit as st
import cv2
import mediapipe as mp
import simpleaudio as sa
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from feedbackbot import bot
from relevancy_checker import check_relevancy
from followup_bot import run_chatbot
import subprocess
import threading
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1)

EXERCISE_GIFS = {
    "Pushups": r"C:\Users\tejat\Desktop\FitnessApp\triangle-pushup.gif",
    #"Squats": "gifs/squats.gif",
    #"Deadlift": "gifs/deadlift.gif",
    #"Skipping": "gifs/skipping.gif",
    #"Plank": "gifs/plank.gif",
    #"Lunges": "gifs/lunges.gif",
    #"High Knees": "gifs/high_knees.gif"
}
def display_exercise_gif(exercise_name):
    gif_path = EXERCISE_GIFS.get(exercise_name)
    
    # Create columns for layout
    col1, col2 = st.columns([4, 1])  # Title takes more space than GIF
    
    with col1:
        st.title("🏋️ Fitness AI Trainer")
        st.markdown("**Real-time exercise analysis** with pose detection and AI-powered feedback.")
    
    with col2:
        if gif_path:
            if gif_path.startswith(('http://', 'https://')):
                # For web-hosted GIFs
                st.markdown(
                    f'<div style="width: 120px; height: 120px; margin-top: 10px;">'
                    f'<img src="{gif_path}" style="width: 100%; height: 100%;">'
                    f'</div>',
                    unsafe_allow_html=True
                )
            elif os.path.exists(gif_path):
                # For local GIF files - read as bytes and display
                with open(gif_path, "rb") as f:
                    gif_bytes = f.read()
                st.image(gif_bytes, width=120, caption="")
            else:
                st.warning("GIF not found")
        else:
            st.warning("No GIF path specified")
def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    ba = a - b
    bc = c - b

    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    if mag_ba * mag_bc < 1e-6:
        return 0

    dot_product = np.dot(ba, bc)
    angle_rad = np.arccos(np.clip(dot_product / (mag_ba * mag_bc), -1.0, 1.0))
    return np.degrees(angle_rad)

class PushupAnalyzer:
    def __init__(self):
        self.prev_elbow_angle = 180  # Start at top position
        self.pushup_count = 0
        self.min_elbow_angle = 180
        self.lockout_threshold = 160
        self.bottom_threshold = 90
        self.is_bottom_reached = False  # Track state internally

    def process_frame(self, shoulder, elbow, wrist, hip):
        pushup_completed = False
        feedback = ""
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        torso_angle = calculate_angle(shoulder, hip, [hip[0], shoulder[1]])

        # Phase 1: Descending (check for bottom)
        if elbow_angle < self.prev_elbow_angle and not self.is_bottom_reached:
            if elbow_angle <= self.bottom_threshold:
                self.is_bottom_reached = True
                self.min_elbow_angle = elbow_angle
                feedback = "Perfect depth!"
            else:
                feedback = f"Go lower! Current: {elbow_angle}°"

        # Phase 2: Ascending (check for top)
        elif elbow_angle > self.prev_elbow_angle and self.is_bottom_reached:
            if elbow_angle >= self.lockout_threshold:
                pushup_completed = True
                self.pushup_count += 1
                self.is_bottom_reached = False
                feedback = "Good rep! " if self.min_elbow_angle <= 90 else "Shallow rep! "
                feedback += "Perfect lockout!" if elbow_angle >= 175 else "Fully extend arms!"
            else:
                feedback = "Keep pushing up!"

        # Torso stability check
        if abs(torso_angle) > 20:
            feedback += " Hips too high!" if torso_angle > 0 else " Hips sagging!"

        self.prev_elbow_angle = elbow_angle
        return pushup_completed, feedback.strip(), elbow_angle, torso_angle
class PiperTTS:
    def __init__(self):
        self.piper_path = r"C:\Users\tejat\Desktop\FitnessApp\piper\piper.exe"
        self.model_path = r"C:\Users\tejat\Desktop\FitnessApp\piper\en_US-kathleen-low.onnx"
        
    def speak(self, text):
        """Non-blocking speech synthesis"""
        def _run():
            output = "feedback.wav"
            cmd = f'echo "{text}" | "{self.piper_path}" --model "{self.model_path}" --output_file {output}'
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
            #playsound(output)
            
        thread = threading.Thread(target=_run)
        thread.start()

# Usage
tts = PiperTTS()


def play_audio_async():
    wave_obj = sa.WaveObject.from_wave_file("feedback.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

def draw_pose(frame, landmarks):
    """Draw landmarks on the frame."""
    mp_drawing.draw_landmarks(
        frame,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
    )

def display_info(frame, pushup_count, feedback, elbow_angle, torso_angle):
    """Display pushup count, angles, and feedback."""
    cv2.putText(frame, f"Pushups: {pushup_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Elbow: {elbow_angle:.1f}°", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Torso: {torso_angle:.1f}°", (50, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if feedback:
        color = (0, 255, 0) if "Perfect" in feedback else (0, 0, 255)
        cv2.putText(frame, feedback, (50, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
def check_relevancy(text1,text2):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([text1,text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()  
    if similarity >= 0.2:
        return "Relevant"
    else:
        return "Irrelevant"
def bot(feedback, exercise):
    

    answer = []
    for f in feedback:
        chat = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,  
            groq_api_key=api_key
        )
        
        # Option 1: Using f-string (no template variables)
        user_message = f"while doing {exercise} this is the feedback I received: {f}. Suggest me some causes, corrections, Risks & Related information and solution progression"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful gym and workout assistant that always answers in following format:\n\n"
            "**Causes:**\n- [List the causes]\n\n"
            "**Corrections:**\n- [List how to prevent it]\n\n"
            "**Risks & Related information:**\n- [List possible Risks and related info]\n\n"
            "**Solution Progression:**\n- [List possible solutions]\n\n"
            "Please follow this structure strictly."),
            ("human", user_message)
        ])
        
        chain = prompt | chat
        response = chain.invoke({"exercise": exercise, "f": f})  # Consistent variable names
        answer.append(response.content)
    
    return answer
def run_chatbot(user_input,text1):
    chat = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7,groq_api_key=api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful fitness assistant...answer the user question based on conversation history {text1}"),
        ("human", "{user_input}")
    ])
    chain = prompt | chat
    response = chain.invoke({"user_input": user_input,"text1":text1})
    return response.content