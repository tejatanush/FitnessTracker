import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1)

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

def evaluate_pushup(shoulder, elbow, wrist, hip, is_down):
    """Evaluate pushup count and form."""
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    torso_angle = calculate_angle(shoulder, hip, [hip[0], shoulder[1]])

    pushup_completed = False
    feedback = ""

    if elbow_angle <= 90 and not is_down:
        is_down = True
    elif elbow_angle >= 160 and is_down:
        pushup_completed = True
        is_down = False

        if torso_angle > 15:
            feedback = "Form Error: Hips sagging!"
        elif elbow_angle > 100:
            feedback = "Form Error: Not low enough!"
        else:
            feedback = "Pushup Perfect!"

    return pushup_completed, is_down, feedback, elbow_angle, torso_angle

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

# Load video
video_path = "pushups2.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video is opened
if not cap.isOpened():
    print("Error: Could not open video file!")
    exit()

pushup_count = 0
is_down = False
current_feedback = ""
feedback_display_frames = 0

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

        # Extract key landmarks
        try:
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]

            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h]

            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h]

            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h]

            # Evaluate pushup
            completed, is_down, feedback, elbow_angle, torso_angle = evaluate_pushup(shoulder, elbow, wrist, hip, is_down)

            if completed:
                pushup_count += 1
                current_feedback = feedback
                feedback_display_frames = 30  # Show feedback for 30 frames

            display_info(frame, pushup_count, 
                         current_feedback if feedback_display_frames > 0 else "",
                         elbow_angle, torso_angle)

            if feedback_display_frames > 0:
                feedback_display_frames -= 1
        except IndexError:
            print("Error: Landmark detection failed!")

    cv2.imshow('Pushup Counter', frame)
    
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
