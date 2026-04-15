# emotion_youtube_streamlit_10sec_final.py
import cv2
import numpy as np
import time
import random
import mediapipe as mp
import streamlit as st
from tensorflow.keras.models import load_model
from youtubesearchpython import VideosSearch

# ----------------------
# CONFIG
# ----------------------
MODEL_PATH = "emotion_model.h5"
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load model safely
model = load_model(MODEL_PATH, compile=False)

# MediaPipe face detection
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)

# ----------------------
# STREAMLIT UI
# ----------------------
st.title("🎵 Emotion  Song")
st.write("This app captures your emotion from the webcam for , "
         "figures out your mood, and plays a matching YouTube song.")

if st.button("📷 Start Detection"):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    last_time = 0
    DETECT_INTERVAL = 1.0

    # Store all detected emotions in this list
    detected_emotions = []

    frame_placeholder = st.empty()
    emotion_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()

        # Detect every DETECT_INTERVAL seconds
        if now - last_time > DETECT_INTERVAL:
            last_time = now
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_face.process(rgb)

            if res.detections:
                for detection in res.detections:
                    d = detection.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x1 = max(int(d.xmin * w), 0)
                    y1 = max(int(d.ymin * h), 0)
                    x2 = min(int((d.xmin + d.width) * w), w - 1)
                    y2 = min(int((d.ymin + d.height) * h), h - 1)

                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # Preprocess face
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48, 48))
                    face_norm = face_resized / 255.0
                    inp = np.expand_dims(np.expand_dims(face_norm, 0), -1)

                    # Predict emotion
                    preds = model.predict(inp, verbose=0)
                    idx = np.argmax(preds)
                    emotion = EMOTION_LABELS[idx]
                    detected_emotions.append(emotion)

                    # Draw on frame
                    cv2.putText(frame, f"{emotion} ({preds[0][idx]*100:.1f}%)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    emotion_text.markdown(f"### Current Emotion: **{emotion}** 😃")

        # Show in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Stop after 10 seconds
        if now - start_time >= 10:
            break

    cap.release()

    # Decide the final emotion (most frequent)
    if detected_emotions:
        final_emotion = max(set(detected_emotions), key=detected_emotions.count)
        st.success(f"✅ Final Detected Emotion: **{final_emotion}**")

        # Search and play song
        search = VideosSearch(f"{final_emotion} mood songs", limit=5)
        results = search.result().get('result', [])
        if results:
            video = random.choice(results)
            link = video['link']
            st.markdown(f"### 🎬 Now Playing: {video['title']}")
            st.video(link)
    else:
        st.error("No face detected. Please try again.")
