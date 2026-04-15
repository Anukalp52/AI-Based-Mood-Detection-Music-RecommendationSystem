import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "emotion_model.h5"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Emotion labels your model produces (adjust if different)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Map emotions to YouTube search keywords
EMOTION_TO_QUERY = {
    "Angry": "anger management music",
    "Disgust": "uplifting songs",
    "Fear": "calm relaxing music",
    "Happy": "happy hits",
    "Neutral": "chill background music",
    "Sad": "sad songs",
    "Surprise": "energetic pop songs"
}

# Load face detector cascade and emotion model
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# EMOTION DETECTION FUNCTION
# -----------------------------
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return frame, None  # No face detected

    emotion_label = None
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)
        if preds is None or len(preds) == 0:
            emotion_label = None
        else:
            emotion_label = EMOTIONS[np.argmax(preds[0])]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        if emotion_label:
            cv2.putText(frame, emotion_label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        break  # Process only first detected face

    return frame, emotion_label

# -----------------------------
# YOUTUBE SEARCH URL GENERATOR
# -----------------------------
def get_youtube_search_url(emotion):
    query = EMOTION_TO_QUERY.get(emotion, "popular music")
    search_query = query.replace(' ', '+')
    url = f"https://www.youtube.com/results?search_query={search_query}"
    return url

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Emotion-Based YouTube Music Recommender", layout="centered")

st.title("😊 Emotion-Based Music Recommendation (YouTube)")
st.markdown(
    "Capture a photo with your webcam or upload an image, then get a YouTube music recommendation based on your detected emotion."
)

# Input options: Webcam photo or upload file
img_file_buffer = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or upload a photo", type=["png", "jpg", "jpeg"])

image = None
if img_file_buffer is not None:
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

if image is not None:
    processed_frame, detected_emotion = process_frame(image)
    st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Processed Image with Detected Emotion")

    if detected_emotion:
        st.write(f"**Detected Emotion:** {detected_emotion}")
        yt_url = get_youtube_search_url(detected_emotion)
        st.markdown(f"### Recommended Music on YouTube: [Click here to listen]({yt_url})")
    else:
        st.warning("No face detected or emotion could not be classified. Please try a different image.")
else:
    st.info("Please capture an image with webcam or upload a photo to begin.")
