# emotion_youtube_player.py
import os, time, random
import cv2, numpy as np, pygame
import mediapipe as mp
from tensorflow.keras.models import load_model
from youtubesearchpython import VideosSearch
import yt_dlp

# ==== CONFIG ====
MODEL_PATH = "emotion_model.h5"
CACHE_DIR = "yt_cache"
FFMPEG_PATH = r"C:\Users\anuka\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"  # change to your ffmpeg path

# ==== PREP ====
os.makedirs(CACHE_DIR, exist_ok=True)
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Load emotion detection model
model = load_model(MODEL_PATH)

# Load Mediapipe face detector
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Init pygame for audio playback
pygame.mixer.init()

# yt-dlp options
YTDL_OPTS = {
    "format": "bestaudio/best",
    "outtmpl": os.path.join(CACHE_DIR, "%(id)s.%(ext)s"),
    "noplaylist": True,
    "quiet": True,
    "ffmpeg_location": FFMPEG_PATH,
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
        "preferredquality": "192",
    }],
}

ydl = yt_dlp.YoutubeDL(YTDL_OPTS)

def download_audio_from_video_id(video_id):
    """Download audio for a given YouTube video ID, return mp3 path."""
    filename_mp3 = os.path.join(CACHE_DIR, video_id + ".mp3")
    if os.path.exists(filename_mp3) and os.path.getsize(filename_mp3) > 10000:
        return filename_mp3
    try:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        if os.path.exists(filename_mp3) and os.path.getsize(filename_mp3) > 10000:
            return filename_mp3
    except Exception as e:
        print("yt-dlp error:", e)
    return None

def play_audio(file_path):
    """Play MP3 file using pygame."""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        print("Audio play error:", e)

# ==== CAMERA LOOP ====
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
last_emotion = None
last_time = 0
DETECT_INTERVAL = 1.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    now = time.time()

    if now - last_time > DETECT_INTERVAL:
        last_time = now
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        if res.detections:
            d = res.detections[0].location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x1 = max(int(d.xmin * w), 0)
            y1 = max(int(d.ymin * h), 0)
            x2 = min(int((d.xmin + d.width) * w), w - 1)
            y2 = min(int((d.ymin + d.height) * h), h - 1)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            inp = cv2.resize(face_gray, (48, 48)) / 255.0
            inp = np.expand_dims(np.expand_dims(inp, 0), -1)

            preds = model.predict(inp, verbose=0)
            eidx = int(np.argmax(preds))
            emotion = emotion_labels[eidx]
            print("Detected Emotion:", emotion)

            # Only change song if emotion changes
            if emotion != last_emotion:
                last_emotion = emotion

                try:
                    vs = VideosSearch(f"{emotion} mood songs", limit=1)
                    resu = vs.result()
                    if resu['result']:
                        vid = resu['result'][0]['id']
                        print("Found video ID:", vid)

                        audio_path = download_audio_from_video_id(vid)
                        if audio_path:
                            play_audio(audio_path)
                        else:
                            print("Audio download failed.")
                except Exception as e:
                    print("YouTube search error:", e)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
