import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time
import serial
import math

# -------- LOAD MODEL --------
model = load_model("../../models/eye_model.keras")
IMG_SIZE = 48

# -------- MEDIAPIPE --------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

MODEL_PATH = "C:\\Users\\reddy\\OneDrive\\Documents\\bus_safety_ai\\src\\data_processing\\face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# -------- SERIAL SETUP --------
try:
    ser = serial.Serial("COM3", 115200, timeout=1)
    print("✅ Serial Connected (IMU + COUNT)")
except:
    ser = None
    print("❌ Serial Not Connected")

# -------- GLOBAL STATES --------
student_count = 0
last_event = ""
last_door = ""

# -------- SERIAL PARSER --------
def parse_serial_line(line):
    global student_count, last_event, last_door

    line = line.strip()

    # ---- COUNT MESSAGE ----
    if line.startswith("COUNT:"):
        try:
            parts = line.split(",")
            student_count = int(parts[0].split(":")[1])

            for p in parts[1:]:
                if p.startswith("DOOR:"):
                    last_door = p.split(":")[1]
                elif p.startswith("EVENT:"):
                    last_event = p.split(":")[1]

            return {"type": "count"}

        except:
            return None

    # ---- IMU MESSAGE ----
    try:
        ax, ay, az, gx, gy, gz = map(float, line.split(","))
        return {"type": "imu", "data": (ax, ay, az, gx, gy, gz)}
    except:
        return None

# -------- IMU PROCESS --------
def process_imu(data):
    if data is None:
        return "unknown"

    ax, ay, az, gx, gy, gz = data

    pitch = math.degrees(math.atan2(ax, az))
    acc_mag = (ax**2 + ay**2 + az**2) ** 0.5

    if acc_mag > 3:
        return "fall"

    if abs(pitch) > 35:
        return "tilt"

    return "normal"

# -------- SEND ALERT --------
def send_alert(msg):
    if ser:
        try:
            ser.write((msg + "\n").encode())
        except:
            pass

# -------- DROWSINESS --------
class DrowsinessTracker:
    def __init__(self):
        self.buffer = deque(maxlen=50)
        self.start_time = None

    def update(self, prob):
        closed = prob > 0.6
        self.buffer.append(closed)

        if closed:
            if self.start_time is None:
                self.start_time = time.time()
        else:
            self.start_time = None

        perclos = sum(self.buffer) / len(self.buffer)

        duration = 0
        if self.start_time:
            duration = time.time() - self.start_time

        score = min(1.0, perclos + duration / 2)
        return score, perclos, duration

tracker = DrowsinessTracker()

# -------- EYE EXTRACTION --------
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

def extract_eye(frame, lm, indices, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]

    x1, x2 = max(min(x)-10, 0), min(max(x)+10, w)
    y1, y2 = max(min(y)-10, 0), min(max(y)+10, h)

    eye = frame[y1:y2, x1:x2]

    if eye is None or eye.size == 0:
        return None

    eye = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = eye / 255.0
    eye = eye.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return eye

# -------- CAMERA --------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera failed")
    exit()

print("✅ FULL SYSTEM STARTED (ML + IMU + COUNT)")

# -------- MAIN LOOP --------
imu_data = None
end_route = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape

    # ---- SERIAL READ ----
    if ser:
        try:
            line = ser.readline().decode(errors="ignore")
            msg = parse_serial_line(line)

            if msg and msg["type"] == "imu":
                imu_data = msg["data"]

        except:
            pass

    # ---- CAMERA PROCESS ----
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    eye_prob = 0.0

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        left_eye = extract_eye(frame, lm, LEFT_EYE, w, h)
        right_eye = extract_eye(frame, lm, RIGHT_EYE, w, h)

        probs = []

        for eye in [left_eye, right_eye]:
            if eye is not None:
                pred = model.predict(eye, verbose=0)[0][0]
                probs.append(pred)

        if probs:
            eye_prob = sum(probs) / len(probs)

    # ---- DROWSINESS ----
    score, perclos, duration = tracker.update(eye_prob)

    # ---- IMU ----
    imu_state = process_imu(imu_data)

    # ---- FUSION ----
    if imu_state == "fall":
        final_alert = "CRITICAL"
    elif score > 0.7 and imu_state == "tilt":
        final_alert = "HIGH"
    elif score > 0.7:
        final_alert = "MEDIUM"
    else:
        final_alert = "NORMAL"

    # ---- CHILD LEFT CHECK ----
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        end_route = True

    if end_route and student_count > 0:
        final_alert = "CHILD LEFT"

    # ---- SEND ALERT ----
    if final_alert == "CRITICAL":
        send_alert("ALERT_CRITICAL")
    elif final_alert == "HIGH":
        send_alert("ALERT_HIGH")
    elif final_alert == "CHILD LEFT":
        send_alert("ALERT_CHILD")

    # ---- COLOR ----
    color = (0,255,0)
    if final_alert in ["CRITICAL", "HIGH", "CHILD LEFT"]:
        color = (0,0,255)
    elif final_alert == "MEDIUM":
        color = (0,255,255)

    # ---- DISPLAY ----
    cv2.putText(frame, f"EyeProb: {eye_prob:.2f}", (10,30), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10,60), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"IMU: {imu_state}", (10,90), 0, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Students: {student_count}", (10,130), 0, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"{last_event} @ {last_door}", (10,160), 0, 0.6, (200,200,255), 2)

    cv2.putText(frame, f"FINAL: {final_alert}", (10,200), 0, 1.0, color, 3)

    cv2.imshow("SMART BUS SAFETY SYSTEM", frame)

    if key == 27:
        break

# -------- CLEANUP --------
cap.release()
cv2.destroyAllWindows()
del landmarker