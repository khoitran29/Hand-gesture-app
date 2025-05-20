#Không phân biệt tay trái tay phải, định dạng lại

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import math

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#--------------------Feature function------------------------
# Calculate Euclidean distance between two landmarks
def landmarks_distance(a, b, img_shape):
    h, w, _ = img_shape
    xa, ya = int(a.x * w), int(a.y * h)
    xb, yb = int(b.x * w), int(b.y * h)
    return np.sqrt((xa - xb)**2 + (ya - yb)**2)

# Count number of raised fingers
def number_identify(hand_landmarks, img_shape):
    lm = hand_landmarks.landmark
    count = 0
    # Thumb (compare with palm landmark)
    if landmarks_distance(lm[4], lm[17], img_shape) > landmarks_distance(lm[2], lm[17], img_shape):
        count += 1
    # Index finger
    if landmarks_distance(lm[8], lm[0], img_shape) > landmarks_distance(lm[5], lm[0], img_shape):
        count += 1
    # Middle finger
    if landmarks_distance(lm[12], lm[0], img_shape) > landmarks_distance(lm[9], lm[0], img_shape):
        count += 1
    # Ring finger
    if landmarks_distance(lm[16], lm[0], img_shape) > landmarks_distance(lm[13], lm[0], img_shape):
        count += 1
    # Pinky finger
    if landmarks_distance(lm[20], lm[0], img_shape) > landmarks_distance(lm[17], lm[0], img_shape):
        count += 1
    return count

# Calculate angle between three landmarks
def landmarks_angular(lm1, lm2, lm3):
    v1 = (lm1.x - lm2.x, lm1.y - lm2.y)
    v2 = (lm3.x - lm2.x, lm3.y - lm2.y)
    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
    mag_v1 = (v1[0]**2 + v1[1]**2) ** 0.5
    mag_v2 = (v2[0]**2 + v2[1]**2) ** 0.5
    
    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0
    
    cos_angle = max(min(dot_product / (mag_v1 * mag_v2), 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

# Extract all gesture features for authentication
def extract_gesture_features(hand_landmarks, img_shape, handedness = None):
    lm = hand_landmarks.landmark
    return {
        'finger_count': number_identify(hand_landmarks, img_shape),
        'thumb_angle': landmarks_angular(lm[0], lm[2], lm[4]),
        'index_angle': landmarks_angular(lm[5], lm[6], lm[8]),
        'middle_angle': landmarks_angular(lm[9], lm[10], lm[12]),
        'ring_angle': landmarks_angular(lm[13], lm[14], lm[16]),
        'pinky_angle': landmarks_angular(lm[17], lm[18], lm[20]),
        'thumb_index_angle': landmarks_angular(lm[4], lm[0], lm[8]),
        'index_middle_angle': landmarks_angular(lm[8], lm[0], lm[12]),
        'middle_ring_angle': landmarks_angular(lm[12], lm[0], lm[16]),
        'ring_pinky_angle': landmarks_angular(lm[16], lm[0], lm[20])
    }

# Compare current gesture with saved password within tolerance
def compare_gestures(current, saved, tolerance=0.5):
    for key in saved:
        if key == 'finger_count':
            if current[key] != saved[key]:
                return False
        else:
            low = saved[key] * (1 - tolerance)
            high = saved[key] * (1 + tolerance)
            if not (low <= current[key] <= high):
                return False
    return True

# -------------------Streamlit UI Setup--------------------------
st.title("Hand Gesture Recognition with MediaPipe")

# Control checkboxes
st.write("Control")
camera_on = st.checkbox("Enable Camera")
set_new_password = st.checkbox("Set Gesture Password")
unlock_by_gesture = st.checkbox("Unlock by Gesture")

# Password recording button
if set_new_password:
    if st.button("Record Gesture Password"):
        st.session_state.record_gesture = True
        st.success("Ready to record gesture - perform your gesture now")
        
# Video display window
FRAME_WINDOW = st.image([])

#Feature checkboxes
st.write("Feature")
counting_number = st.checkbox("Show Finger Counting")
display_landmarks = st.checkbox("Show Hand Landmarks")
show_features = st.checkbox("Show Feature Values")

# Initialize session state variables
if 'gesture_password' not in st.session_state:
    st.session_state.gesture_password = None
if 'record_gesture' not in st.session_state:
    st.session_state.record_gesture = False
if 'last_unlock_time' not in st.session_state:
    st.session_state.last_unlock_time = 0

# Background
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(120deg, #f6d365 20%, #fda085 100%);
        background-size: cover;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------Main processing loop--------------------------
if camera_on:
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        total_fingers = 0
        gesture_info = None

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:
                # Finger counting display
                if counting_number:
                    fingers = number_identify(hand_landmarks, frame.shape)
                    total_fingers += fingers
                    cv2.putText(frame, f"Fingers: {total_fingers}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Draw hand landmarks
                if display_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract current gesture features
                gesture_info = extract_gesture_features(hand_landmarks, frame.shape)

                # Record new gesture password
                if st.session_state.record_gesture:
                    st.session_state.gesture_password = gesture_info
                    st.session_state.record_gesture = False
                    st.success("Gesture password saved!")
                    st.write("Saved features:", st.session_state.gesture_password)

                # Gesture authentication logic
                current_time = time.time()
                if (unlock_by_gesture and 
                    st.session_state.gesture_password and 
                    (current_time - st.session_state.last_unlock_time) > 3):
                    
                    if compare_gestures(gesture_info, st.session_state.gesture_password):
                        st.session_state.last_unlock_time = current_time

                # Display feature values
                if show_features and gesture_info:
                    y_pos = 10
                    for key, val in gesture_info.items():
                        text = f"{key}: {val:.2f}" if isinstance(val, float) else f"{key}: {val}"
                        cv2.putText(frame, text, (450, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        y_pos += 15

        # Show unlock message for 3 seconds after successful authentication
        if (time.time() - st.session_state.last_unlock_time) <= 3:
            cv2.putText(frame, "UNLOCKED", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.write("Camera is disabled")