#!/usr/bin/env python3
"""
Webcam-based emoji detector for 3 expressions:
- Surprised (mouth open)
- One hand up
- Thinking (head tilt)
"""

import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Configuration
FREAKY_THRESHOLD = 0.55
WINDOW_WIDTH, WINDOW_HEIGHT = 720, 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Load emojis
emoji_files = {
    "FREAKY": "freaky.jpeg",
    "ONE_HAND_UP": "monkeygotit.jpeg",
    "THINKING": "monkeythinking.jpg"
}

emojis = {}
for key, path in emoji_files.items():
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Missing image: {path}")
        exit()
    emojis[key] = cv2.resize(img, EMOJI_WINDOW_SIZE)

blank_emoji = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)

print("Press 'q' to quit.\n")

# Main loop
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        state = "THINKING"  # Default

        # Pose detection
        pose_results = pose.process(rgb)
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            left_shoulder, right_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist, right_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST], lm[mp_pose.PoseLandmark.RIGHT_WRIST]

            left_up = left_wrist.y < left_shoulder.y
            right_up = right_wrist.y < right_shoulder.y

            # One hand up
            if (left_up and not right_up) or (right_up and not left_up):
                state = "ONE_HAND_UP"

            # Thinking (head tilt)
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            if shoulder_diff > 0.1:
                state = "THINKING"

        # Face mesh for FREAKY
        face_results = face_mesh.process(rgb)
        if face_results.multi_face_landmarks:
            for f in face_results.multi_face_landmarks:
                left_corner, right_corner = f.landmark[291], f.landmark[61]
                upper_lip, lower_lip = f.landmark[13], f.landmark[14]
                mouth_width = math.hypot(right_corner.x - left_corner.x, right_corner.y - left_corner.y)
                mouth_height = math.hypot(lower_lip.x - upper_lip.x, lower_lip.y - upper_lip.y)
                if mouth_width > 0:
                    ratio = mouth_height / mouth_width
                    if ratio > FREAKY_THRESHOLD:
                        state = "FREAKY"

        # Select emoji and text
        emoji = emojis.get(state, blank_emoji)
        emoji_symbol = {"FREAKY": "😝", "ONE_HAND_UP": "☝️", "THINKING": "🤔"}.get(state, "❓")

        frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.putText(frame_resized, f"{state} {emoji_symbol}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Camera Feed', frame_resized)
        cv2.imshow('Emoji Output', emoji)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
