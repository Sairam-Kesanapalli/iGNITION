import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# -------- CONFIG --------
BASE_DIR = "../../data/head_dataset"

labels = {
    "1": "normal",
    "2": "tilt_forward",
    "4": "tilt_side",
    "3": "look_away"
}

current_label = "normal"
recording = False
writer = None
file = None

# -------- MEDIAPIPE --------
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

MODEL_PATH = "C:\\Users\\reddy\\OneDrive\\Documents\\bus_safety_ai\\src\\data_processing\\face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# -------- CAMERA --------
cap = cv2.VideoCapture(0)

print("""
Controls:
1 → NORMAL
2 → TILT
3 → LOOK_AWAY
r → START recording
s → STOP recording
ESC → EXIT
""")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    pitch, yaw, roll = 0, 0, 0

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        nose = lm[1]
        chin = lm[152]
        le = lm[33]
        re = lm[263]

        nose = np.array([nose.x*w, nose.y*h])
        chin = np.array([chin.x*w, chin.y*h])
        le = np.array([le.x*w, le.y*h])
        re = np.array([re.x*w, re.y*h])

        pitch = np.degrees(np.arctan2(chin[1]-nose[1], chin[0]-nose[0]))
        yaw = nose[0] - (le[0] + re[0]) / 2
        roll = np.degrees(np.arctan2(re[1]-le[1], re[0]-le[0]))

        # -------- SAVE DATA --------
        if recording and writer:
            writer.writerow([pitch, yaw, roll])

    # -------- DISPLAY --------
    status = "RECORDING" if recording else "IDLE"

    cv2.putText(frame, f"Class: {current_label}", (10,30), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Status: {status}", (10,60), 0, 0.7, (0,255,0) if recording else (0,0,255), 2)

    cv2.putText(frame, "1:Normal  2:ForwardTilt  3:SideTilt  4:LookAway", (10,100),
            0, 0.6, (200,200,255), 1)
    cv2.putText(frame, "r:Start  s:Stop  ESC:Exit", (10,130), 0, 0.6, (200,200,255), 1)

    cv2.imshow("Head Data Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------- CLASS SELECTION --------
    if chr(key) in labels:
        current_label = labels[chr(key)]
        print("Selected:", current_label)

    # -------- START RECORDING --------
    elif key == ord('r') and not recording:
        folder = os.path.join(BASE_DIR, current_label)
        os.makedirs(folder, exist_ok=True)

        filename = f"{current_label}_{int(time.time())}.csv"
        filepath = os.path.join(folder, filename)

        file = open(filepath, "w", newline="")
        writer = csv.writer(file)
        writer.writerow(["pitch", "yaw", "roll"])

        recording = True
        print("Recording started:", filepath)

    # -------- STOP RECORDING --------
    elif key == ord('s') and recording:
        recording = False
        if file:
            file.close()
        writer = None
        print("Recording stopped")

    # -------- EXIT --------
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
del landmarker