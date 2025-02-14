import cv2
import time
import threading
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Define eye landmarks
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 387, 373]

# Variables for threading
last_capture_time = time.time()
capture_interval = 5  # Run emotion detection every 5 seconds
emotion_result = "Detecting..."
processing_emotion = False  # Flag to check if thread is running

def analyze_emotion(frame):
    """Runs DeepFace emotion detection in a separate thread."""
    global emotion_result, processing_emotion
    processing_emotion = True  # Mark as processing
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion_result = result[0]['dominant_emotion']
    except Exception as e:
        emotion_result = "Error"
    processing_emotion = False  # Done processing

def get_eye_region(landmarks, eye_points, frame):
    """ Extracts the eye region using facial landmarks """
    h, w, _ = frame.shape
    x_coords = [int(landmarks.landmark[i].x * w) for i in eye_points]
    y_coords = [int(landmarks.landmark[i].y * h) for i in eye_points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    eye_region = frame[y_min:y_max, x_min:x_max]
    return eye_region, x_min, y_min

def detect_pupil(eye_region):
    """ Detects the pupil using thresholding & contours. """
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)  # Reduce noise
    _, thresholded = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)  # Invert colors
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get largest contour (assumed to be the pupil)
        pupil_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(pupil_contour)

        if M["m00"] != 0:
            px = int(M["m10"] / M["m00"])
            py = int(M["m01"] / M["m00"])
            return px, py  # Return pupil center

    return None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face landmarks
    results = face_mesh.process(rgb_frame)

    gaze_direction = "Unknown"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract left and right eye regions
            left_eye_region, left_x_min, left_y_min = get_eye_region(face_landmarks, LEFT_EYE, frame)
            right_eye_region, right_x_min, right_y_min = get_eye_region(face_landmarks, RIGHT_EYE, frame)

            # Detect pupil positions
            left_px, left_py = detect_pupil(left_eye_region)
            right_px, right_py = detect_pupil(right_eye_region)

            if left_px is not None and right_px is not None:
                # Draw pupils on the original frame
                cv2.circle(frame, (left_x_min + left_px, left_y_min + left_py), 3, (0, 255, 0), -1)
                cv2.circle(frame, (right_x_min + right_px, right_y_min + right_py), 3, (0, 255, 0), -1)

                # Normalize pupil position
                left_pupil_x_ratio = left_px / left_eye_region.shape[1]
                right_pupil_x_ratio = right_px / right_eye_region.shape[1]

                left_pupil_y_ratio = left_py / left_eye_region.shape[0]
                right_pupil_y_ratio = right_py / right_eye_region.shape[0]

                # Detect horizontal gaze
                if left_pupil_x_ratio < 0.40 and right_pupil_x_ratio < 0.40:
                    gaze_direction = "Looking Left"
                elif left_pupil_x_ratio > 0.60 and right_pupil_x_ratio > 0.60:
                    gaze_direction = "Looking Right"
                else:
                    gaze_direction = "Looking Center"

                # Detect vertical gaze
                if left_pupil_y_ratio < 0.40 and right_pupil_y_ratio < 0.40:
                    gaze_direction = "Looking Up"
                elif left_pupil_y_ratio > 0.60 and right_pupil_y_ratio > 0.60:
                    gaze_direction = "Looking Down"

    # Check if it's time to analyze emotion
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval and not processing_emotion:
        last_capture_time = current_time  # Reset timer
        threading.Thread(target=analyze_emotion, args=(frame.copy(),), daemon=True).start()

    # Display gaze direction & emotion result
    cv2.putText(frame, f"Gaze: {gaze_direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Emotion: {emotion_result}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video feed
    cv2.imshow("Pupil Tracking & Emotion Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
