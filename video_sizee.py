import cv2

# Input and output video paths
input_video_path = r"C:\Users\tejat\Desktop\FitnessApp\pushups.mp4"  # Change this to your video file
output_video_path = "output_680x680.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
frame_width = 680
frame_height = 680

# Define the video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break if no more frames

    # Resize frame to 680x680
    resized_frame = cv2.resize(frame, (frame_width, frame_height))

    # Write resized frame to output video
    out.write(resized_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video resized and saved successfully!")
