import streamlit as st
import cv2
import tensorflow as tf
import tf_keras
from collections import deque
import numpy as np
from utils import char_to_num, num_to_char
from modelutil import load_model
import time
import os

# Set page config
st.set_page_config(page_title="Real-Time Lip Reading", layout="wide")

# Title
st.title("🎥 Real-Time Lip Reading")
st.markdown("---")

# Load model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Load Haar Cascade classifiers for face and mouth detection
@st.cache_resource
def get_face_mouth_detectors():
    cascade_path = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_smile.xml')
    return face_cascade, mouth_cascade

face_cascade, mouth_cascade = get_face_mouth_detectors()

# Parameters
FRAME_BUFFER_SIZE = 75  # Number of frames to keep
TARGET_HEIGHT = 46
TARGET_WIDTH = 140
CROP_Y_START = 190
CROP_Y_END = 236
CROP_X_START = 80
CROP_X_END = 220

# Sidebar controls
st.sidebar.header("Controls")
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2], index=0)
prediction_interval = st.sidebar.slider("Prediction Interval (seconds)", 1, 10, 3)
show_processed = st.sidebar.checkbox("Show Processed Frames", value=False)
use_auto_detect = st.sidebar.checkbox("Auto-detect Mouth Region", value=True, help="Automatically detect mouth using face detection")

if not use_auto_detect:
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Manual Mouth Region")
    st.sidebar.caption("Adjust these to center on your mouth")
    crop_y_start = st.sidebar.slider("Top Edge", 0, 400, 190, 10)
    crop_y_end = st.sidebar.slider("Bottom Edge", 0, 400, 236, 10)
    crop_x_start = st.sidebar.slider("Left Edge", 0, 600, 80, 10)
    crop_x_end = st.sidebar.slider("Right Edge", 0, 600, 220, 10)
else:
    # Default values when auto-detect is on
    crop_y_start, crop_y_end = 190, 236
    crop_x_start, crop_x_end = 80, 220

# Buttons
col1, col2 = st.columns(2)
start_button = col1.button("▶️ Start Camera", use_container_width=True)
stop_button = col2.button("⏹️ Stop Camera", use_container_width=True)

# Placeholders
video_placeholder = st.empty()
prediction_placeholder = st.empty()
status_placeholder = st.empty()

# Session state
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'frame_buffer' not in st.session_state:
    st.session_state.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

def detect_mouth_region(frame):
    """Detect face and mouth region using Haar cascades"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) == 0:
        return None, None
    
    # Use the largest face (closest to camera)
    face = max(faces, key=lambda rect: rect[2] * rect[3])
    (fx, fy, fw, fh) = face
    
    # Define mouth region as lower half of face
    # Mouth is typically in bottom 40% of face, centered horizontally
    mouth_y_start = fy + int(fh * 0.6)  # Start at 60% down the face
    mouth_y_end = fy + fh  # End at bottom of face
    mouth_x_start = fx + int(fw * 0.2)  # Start at 20% from left
    mouth_x_end = fx + int(fw * 0.8)  # End at 80% from right
    
    # Ensure coordinates are within frame bounds
    h, w = frame.shape[:2]
    mouth_y_start = max(0, min(mouth_y_start, h))
    mouth_y_end = max(0, min(mouth_y_end, h))
    mouth_x_start = max(0, min(mouth_x_start, w))
    mouth_x_end = max(0, min(mouth_x_end, w))
    
    return (mouth_x_start, mouth_y_start, mouth_x_end, mouth_y_end), face

def preprocess_frame(frame, y_start, y_end, x_start, x_end):
    """Preprocess a single frame for the model"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Validate coordinates
    h, w = gray.shape
    if y_start >= y_end or x_start >= x_end or y_end > h or x_end > w:
        # Invalid coordinates, return None
        return None
    
    # Crop to mouth region
    cropped = gray[y_start:y_end, x_start:x_end]
    
    # Check if crop is valid
    if cropped.size == 0:
        return None
    
    # Ensure correct size
    if cropped.shape[0] != TARGET_HEIGHT or cropped.shape[1] != TARGET_WIDTH:
        cropped = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT))
    
    # Add channel dimension
    cropped = np.expand_dims(cropped, axis=-1)
    
    return cropped

def prepare_frames_for_model(frames):
    """Prepare buffered frames for model prediction"""
    if len(frames) < FRAME_BUFFER_SIZE:
        # Pad with last frame if not enough frames
        while len(frames) < FRAME_BUFFER_SIZE:
            frames.append(frames[-1] if frames else np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 1)))
    
    # Convert to tensor
    frames_tensor = tf.convert_to_tensor(list(frames), dtype=tf.float32)
    
    # Normalize
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)
    normalized = (frames_tensor - mean) / (std + 1e-8)
    
    return normalized

def predict_from_buffer(frames):
    """Make prediction from frame buffer"""
    try:
        # Prepare frames
        processed_frames = prepare_frames_for_model(frames)
        
        # Add batch dimension
        batch = tf.expand_dims(processed_frames, axis=0)
        
        # Predict
        yhat = model.predict(batch)
        
        # Decode prediction
        decoded = tf_keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
        
        # Convert to text
        text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')
        
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# Handle start button
if start_button:
    st.session_state.camera_running = True
    st.session_state.frame_buffer.clear()

# Handle stop button
if stop_button:
    st.session_state.camera_running = False

# Main camera loop
if st.session_state.camera_running:
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        status_placeholder.error("❌ Could not open camera. Please check camera permissions and try a different camera index.")
        st.session_state.camera_running = False
    else:
        status_placeholder.success("✅ Camera is running. Position your face so the mouth is visible.")
        
        last_prediction_time = time.time()
        frame_count = 0
        mouth_detected_count = 0
        
        # Display info
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("Frames Captured", "0")
        info_col2.metric("Mouth Detected", "0")
        info_col3.metric("Last Prediction", "Waiting...")
        
        while st.session_state.camera_running:
            ret, frame = cap.read()
            
            if not ret:
                status_placeholder.warning("⚠️ Failed to capture frame")
                break
            
            # Auto-detect mouth region if enabled
            if use_auto_detect:
                mouth_coords, face_coords = detect_mouth_region(frame)
                if mouth_coords:
                    crop_x_start, crop_y_start, crop_x_end, crop_y_end = mouth_coords
                    mouth_detected_count += 1
                else:
                    # No face detected - skip this frame
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "No face detected - please face the camera", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    video_placeholder.image(display_frame, channels="BGR", use_container_width=True)
                    time.sleep(0.033)
                    continue
            
            # Preprocess and add to buffer
            try:
                processed_frame = preprocess_frame(frame, crop_y_start, crop_y_end, crop_x_start, crop_x_end)
                if processed_frame is not None:
                    st.session_state.frame_buffer.append(processed_frame)
                    frame_count += 1
                else:
                    # Skip invalid frame
                    continue
            except Exception as e:
                status_placeholder.error(f"Error processing frame: {str(e)}")
                continue
            
            # Display frame
            if show_processed:
                # Show the cropped mouth region
                display_frame = cv2.cvtColor(processed_frame.squeeze(), cv2.COLOR_GRAY2BGR)
                display_frame = cv2.resize(display_frame, (420, 138))  # Scale up for visibility
            else:
                # Show original frame with rectangles
                display_frame = frame.copy()
                
                # Draw face rectangle if auto-detect is on
                if use_auto_detect and mouth_coords and face_coords is not None:
                    fx, fy, fw, fh = face_coords
                    cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
                    cv2.putText(display_frame, "Face", (fx, fy - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Draw mouth rectangle
                cv2.rectangle(display_frame, 
                            (crop_x_start, crop_y_start), 
                            (crop_x_end, crop_y_end), 
                            (0, 255, 0), 2)
                cv2.putText(display_frame, "Mouth Region", 
                          (crop_x_start, crop_y_start - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            video_placeholder.image(display_frame, channels="BGR", use_container_width=True)
            
            # Make prediction at intervals
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:
                if len(st.session_state.frame_buffer) >= FRAME_BUFFER_SIZE:
                    with st.spinner("🔍 Reading lips..."):
                        prediction = predict_from_buffer(list(st.session_state.frame_buffer))
                        prediction_placeholder.markdown(f"### 💬 Prediction: `{prediction}`")
                        info_col3.metric("Last Prediction", prediction)
                    last_prediction_time = current_time
                else:
                    prediction_placeholder.info(f"⏳ Collecting frames... ({len(st.session_state.frame_buffer)}/{FRAME_BUFFER_SIZE})")
            
            info_col1.metric("Frames Captured", str(frame_count))
            info_col2.metric("Mouth Detected", str(mouth_detected_count))
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.033)  # ~30 FPS
            
            # Check if stop was pressed
            if not st.session_state.camera_running:
                break
        
        cap.release()
        status_placeholder.info("📷 Camera stopped")
else:
    st.info("👆 Click 'Start Camera' to begin real-time lip reading")
    st.markdown("""
    ### 📋 Instructions:
    1. **Enable Auto-detect** in the sidebar (recommended) - the app will find your mouth automatically
    2. Click **Start Camera** to activate your webcam
    3. **Face the camera directly** - you'll see a blue box around your face and green box around mouth
    4. If no face is detected, adjust lighting and move closer to the camera
    5. Speak clearly and the model will predict what you're saying every few seconds
    6. Click **Stop Camera** when done
    
    ### 💡 Tips for Better Results:
    - **Good lighting** is critical - face a window or lamp
    - Keep your **face centered** and fill most of the frame
    - **Sit still** during the 3-second prediction window
    - Speak **short simple phrases** (3-5 words)
    - The model works best with **clear, exaggerated lip movements**
    - If auto-detect fails, disable it and use manual sliders
    
    ### ⚠️ Important Limitations:
    - The model was trained on specific videos and **won't generalize well** to new speakers
    - Predictions are based on the **last 75 frames** (~2.5 seconds at 30fps)
    - Accuracy depends heavily on matching the training data conditions
    - This is a **proof-of-concept**, not production-ready
    
    ### ⚙️ Adjustable Settings:
    - **Auto-detect Mouth Region**: Let face detection find your mouth (recommended)
    - **Select Camera**: Choose which camera to use (0 is usually the default)
    - **Prediction Interval**: How often to generate predictions (in seconds)
    - **Show Processed Frames**: Toggle between original video and processed mouth region
    - **Manual Mouth Region**: If auto-detect fails, manually adjust the crop area
    """)

st.markdown("---")
st.caption("Real-Time Lip Reading powered by LipNet • Note: Best results with training-set videos")
