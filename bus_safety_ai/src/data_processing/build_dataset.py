import cv2
import mediapipe as mp
import os

# -------- CONFIG --------
VIDEO_DIR = "../../data/videos"
OUTPUT_DIR = "../../data/dataset"

os.makedirs(os.path.join(OUTPUT_DIR, "open"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "closed"), exist_ok=True)

FRAME_SKIP = 5
IMG_SIZE = 48

# -------- MEDIAPIPE --------
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

MODEL_PATH = "C:\\Users\\reddy\\OneDrive\\Documents\\bus_safety_ai\\src\\data_processing\\face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# Eye landmark indices
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

# -------- UTIL --------
def extract_eye(frame, landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]

    x1, x2 = max(min(x)-5, 0), min(max(x)+5, w)
    y1, y2 = max(min(y)-5, 0), min(max(y)+5, h)

    eye = frame[y1:y2, x1:x2]

    if eye is None or eye.size == 0:
        return None

    eye = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

    return eye

# -------- MAIN --------
counter = 0
last_timestamp = -1
global_time_offset = 0

for label in ["open", "closed"]:
    folder = os.path.join(VIDEO_DIR, label)

    for video in os.listdir(folder):
        video_path = os.path.join(folder, video)
        cap = cv2.VideoCapture(video_path)

        print(f"Processing: {video_path}")

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to reduce redundancy
            if frame_id % FRAME_SKIP != 0:
                frame_id += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # ---- TIMESTAMP FIX ----
            local_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = global_time_offset + local_time

            if timestamp <= last_timestamp:
                timestamp = last_timestamp + 1

            last_timestamp = timestamp
            # ------------------------

            result = landmarker.detect_for_video(mp_image, timestamp)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                h, w, _ = frame.shape

                left = extract_eye(frame, lm, LEFT_EYE, w, h)
                right = extract_eye(frame, lm, RIGHT_EYE, w, h)

                for eye in [left, right]:
                    if eye is not None:
                        filename = f"{label}_{counter}.png"
                        save_path = os.path.join(OUTPUT_DIR, label, filename)

                        cv2.imwrite(save_path, eye)
                        counter += 1

            frame_id += 1

        cap.release()

        # Move global time forward for next video
        global_time_offset = last_timestamp + 100

print("Dataset creation complete.")