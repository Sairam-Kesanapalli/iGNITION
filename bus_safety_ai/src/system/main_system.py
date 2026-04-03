import math
import os
import platform
import sys
import time
import json
from collections import deque
from pathlib import Path

import cv2
import keras
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from communication.serial_reader import BusSerialReader
from communication.serial_writer import BusSerialWriter

# -------- PATH CONFIG --------
EYE_MODEL_PATH = Path(os.getenv("EYE_MODEL_PATH", ROOT_DIR / "models" / "eye_model.keras"))
FACE_LANDMARKER_PATH = Path(
    os.getenv(
        "FACE_LANDMARKER_PATH",
        ROOT_DIR / "src" / "data_processing" / "face_landmarker.task",
    )
)

SERIAL_PORT = os.getenv("BUS_SERIAL_PORT", "/dev/ttyUSB0")
SERIAL_BAUD = int(os.getenv("BUS_SERIAL_BAUD", "115200"))
CAMERA_INDEX = int(os.getenv("BUS_CAMERA_INDEX", "0"))
IMG_SIZE = 48
EYE_MODEL_OUTPUT = os.getenv("EYE_MODEL_OUTPUT", "open").strip().lower()
MODEL_CLOSED_THRESHOLD = float(os.getenv("MODEL_CLOSED_THRESHOLD", "0.45"))
EAR_CLOSED_THRESHOLD = float(os.getenv("EAR_CLOSED_THRESHOLD", "0.20"))
MEDIUM_ALERT_THRESHOLD = float(os.getenv("MEDIUM_ALERT_THRESHOLD", "0.55"))
HIGH_ALERT_THRESHOLD = float(os.getenv("HIGH_ALERT_THRESHOLD", "0.75"))
AUTO_CALIBRATE = os.getenv("AUTO_CALIBRATE", "1").strip().lower() in {"1", "true", "yes", "on"}
CALIBRATION_SECONDS = int(os.getenv("CALIBRATION_SECONDS", "30"))
CALIBRATION_CLOSE_SECONDS = int(os.getenv("CALIBRATION_CLOSE_SECONDS", "10"))
CALIBRATION_MODE = os.getenv("CALIBRATION_MODE", "full").strip().lower()
HEAD_YAW_THRESHOLD = float(os.getenv("HEAD_YAW_THRESHOLD", "0.22"))
HEAD_PITCH_THRESHOLD = float(os.getenv("HEAD_PITCH_THRESHOLD", "0.20"))
HEAD_ROLL_THRESHOLD_DEG = float(os.getenv("HEAD_ROLL_THRESHOLD_DEG", "22"))
DRIVER_PROFILE_NAME = os.getenv("DRIVER_PROFILE", "sairam").strip().lower()
PROFILE_TARGET_DAYS = int(os.getenv("PROFILE_TARGET_DAYS", "5"))
PROFILE_ROLLING_SESSIONS = int(os.getenv("PROFILE_ROLLING_SESSIONS", "10"))
PROFILE_DIR = ROOT_DIR / "data" / "profiles"
INFERENCE_EVERY_N_FRAMES = max(1, int(os.getenv("INFERENCE_EVERY_N_FRAMES", "3")))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
ALERT_RESEND_INTERVAL_SEC = float(os.getenv("ALERT_RESEND_INTERVAL_SEC", "0.7"))
BUZZER_ON_MEDIUM = os.getenv("BUZZER_ON_MEDIUM", "1").strip().lower() in {"1", "true", "yes", "on"}
MEDIUM_BUZZER_DELAY_SEC = float(os.getenv("MEDIUM_BUZZER_DELAY_SEC", "1.2"))
WINDOW_NAME = "SMART BUS SAFETY SYSTEM"


def _profile_path(driver_name):
    return PROFILE_DIR / f"{driver_name}.json"


def load_driver_profile(driver_name):
    path = _profile_path(driver_name)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Profile load failed ({path}): {exc}")
        return None


def apply_driver_profile(profile):
    global MODEL_CLOSED_THRESHOLD, EAR_CLOSED_THRESHOLD
    thresholds = profile.get("active_thresholds", {})
    model_thr = thresholds.get("model_closed_threshold")
    ear_thr = thresholds.get("ear_closed_threshold")

    if model_thr is not None:
        MODEL_CLOSED_THRESHOLD = float(model_thr)
    if ear_thr is not None:
        EAR_CLOSED_THRESHOLD = float(ear_thr)

    print(
        f"Loaded driver profile '{profile.get('driver_name', 'unknown')}' "
        f"with MODEL_CLOSED_THRESHOLD={MODEL_CLOSED_THRESHOLD:.3f}, "
        f"EAR_CLOSED_THRESHOLD={EAR_CLOSED_THRESHOLD:.3f}"
    )


def save_driver_profile(driver_name, model_thr, ear_thr, metadata=None):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(driver_name)
    now_ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    profile = load_driver_profile(driver_name)
    if profile is None:
        profile = {
            "driver_name": driver_name,
            "created_at": now_ts,
            "target_calibration_days": PROFILE_TARGET_DAYS,
            "sessions": [],
            "active_thresholds": {},
        }

    sessions = profile.get("sessions", [])
    new_session = {
        "timestamp": now_ts,
        "model_closed_threshold": float(model_thr),
        "ear_closed_threshold": float(ear_thr),
    }
    if metadata:
        new_session.update(metadata)
    sessions.append(new_session)

    # Keep bounded history while preserving the most recent sessions.
    sessions = sessions[-60:]

    rolling = sessions[-max(1, PROFILE_ROLLING_SESSIONS):]
    model_values = [float(s["model_closed_threshold"]) for s in rolling if "model_closed_threshold" in s]
    ear_values = [float(s["ear_closed_threshold"]) for s in rolling if "ear_closed_threshold" in s]

    if model_values:
        active_model = float(np.mean(model_values))
    else:
        active_model = float(model_thr)
    if ear_values:
        active_ear = float(np.mean(ear_values))
    else:
        active_ear = float(ear_thr)

    unique_days = sorted({s.get("timestamp", "")[:10] for s in sessions if s.get("timestamp")})

    profile["sessions"] = sessions
    profile["updated_at"] = now_ts
    profile["calibrated_days_count"] = len(unique_days)
    profile["active_thresholds"] = {
        "model_closed_threshold": active_model,
        "ear_closed_threshold": active_ear,
        "computed_from_sessions": len(rolling),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(
        f"Saved driver profile '{driver_name}' -> "
        f"active MODEL={active_model:.3f}, EAR={active_ear:.3f}, "
        f"days={profile['calibrated_days_count']}/{profile.get('target_calibration_days', PROFILE_TARGET_DAYS)}"
    )


def _patch_dense_init(dense_cls):
    if getattr(dense_cls, "_patched_for_quantization_config", False):
        return

    original_init = dense_cls.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        return original_init(self, *args, **kwargs)

    dense_cls.__init__ = patched_init
    dense_cls._patched_for_quantization_config = True


_patch_dense_init(keras.layers.Dense)

# -------- LOAD MODEL --------
model = load_model(
    str(EYE_MODEL_PATH),
    compile=False,
)

# -------- MEDIAPIPE --------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
    num_faces=1,
)

landmarker = FaceLandmarker.create_from_options(options)

# -------- SERIAL SETUP --------
serial_reader = BusSerialReader(port=SERIAL_PORT, baud=SERIAL_BAUD, timeout=1.0)
if serial_reader.connect():
    print(f"Serial connected on {SERIAL_PORT} @ {SERIAL_BAUD}")
else:
    print(f"Serial not connected (port={SERIAL_PORT})")

serial_writer = BusSerialWriter(serial_reader.ser)

# -------- GLOBAL STATES --------
student_count = 0
last_event = ""
last_door = ""
last_distance = None
last_bpm = None

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

# -------- DROWSINESS --------
class DrowsinessTracker:
    def __init__(self):
        self.buffer = deque(maxlen=50)
        self.start_time = None

    def update(self, closed_score, face_detected=True):
        if not face_detected:
            self.buffer.append(0.0)
            self.start_time = None
            perclos = 0.0
            duration = 0.0
            return 0.0, perclos, duration

        closed_score = float(np.clip(closed_score, 0.0, 1.0))
        self.buffer.append(closed_score)
        closed = closed_score >= MODEL_CLOSED_THRESHOLD

        if closed:
            if self.start_time is None:
                self.start_time = time.time()
        else:
            self.start_time = None

        perclos = sum(1.0 for s in self.buffer if s >= MODEL_CLOSED_THRESHOLD) / len(self.buffer)
        avg_closed_score = sum(self.buffer) / len(self.buffer)

        duration = 0
        if self.start_time:
            duration = time.time() - self.start_time

        score = min(1.0, (0.60 * perclos) + (0.25 * avg_closed_score) + (0.15 * min(1.0, duration / 2.0)))
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


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def eye_aspect_ratio(lm, indices, w, h):
    pts = [(float(lm[i].x * w), float(lm[i].y * h)) for i in indices]
    horizontal = _distance(pts[0], pts[3])
    if horizontal <= 1e-6:
        return None
    vertical = _distance(pts[1], pts[5]) + _distance(pts[2], pts[4])
    return vertical / (2.0 * horizontal)


def _safe_point(lm, idx):
    p = lm[idx]
    return float(p.x), float(p.y)


def estimate_head_pose(lm):
    # Mediapipe canonical landmark IDs for coarse pose estimation.
    left_eye_outer = _safe_point(lm, 33)
    right_eye_outer = _safe_point(lm, 263)
    nose_tip = _safe_point(lm, 1)
    mouth_center = _safe_point(lm, 13)

    eye_mid_x = (left_eye_outer[0] + right_eye_outer[0]) / 2.0
    eye_mid_y = (left_eye_outer[1] + right_eye_outer[1]) / 2.0
    eye_dx = right_eye_outer[0] - left_eye_outer[0]
    eye_dy = right_eye_outer[1] - left_eye_outer[1]
    eye_dist = max(1e-6, (eye_dx ** 2 + eye_dy ** 2) ** 0.5)

    # Positive yaw means head turned to driver's right in image space.
    yaw = (nose_tip[0] - eye_mid_x) / eye_dist

    # Pitch is normalized nose vertical displacement between eye-line and mouth.
    face_vertical = max(1e-6, abs(mouth_center[1] - eye_mid_y))
    expected_nose_y = (eye_mid_y + mouth_center[1]) / 2.0
    pitch = (nose_tip[1] - expected_nose_y) / face_vertical

    roll_deg = math.degrees(math.atan2(eye_dy, eye_dx))

    is_frontal = (
        abs(yaw) <= HEAD_YAW_THRESHOLD
        and abs(pitch) <= HEAD_PITCH_THRESHOLD
        and abs(roll_deg) <= HEAD_ROLL_THRESHOLD_DEG
    )

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll_deg": roll_deg,
        "is_frontal": is_frontal,
    }


def closed_prob_from_model(raw_prob):
    raw_prob = float(np.clip(raw_prob, 0.0, 1.0))
    if EYE_MODEL_OUTPUT == "closed":
        return raw_prob
    return 1.0 - raw_prob


def closed_prob_from_ear(ear):
    if ear is None:
        return None
    if EAR_CLOSED_THRESHOLD <= 1e-6:
        return 0.0
    scale = EAR_CLOSED_THRESHOLD * 0.35
    if scale <= 1e-6:
        scale = 0.05
    return float(np.clip((EAR_CLOSED_THRESHOLD - ear) / scale, 0.0, 1.0))


def extract_eye_metrics(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    eye_prob = 0.0
    closed_prob_model = 0.0
    closed_prob_ear = None
    fused_closed_prob = 0.0
    mean_ear = None
    face_detected = False
    head_pose = None
    drowsy_valid = False

    if result.face_landmarks:
        face_detected = True
        lm = result.face_landmarks[0]
        head_pose = estimate_head_pose(lm)
        drowsy_valid = head_pose["is_frontal"]

        left_eye = extract_eye(frame, lm, LEFT_EYE, w, h)
        right_eye = extract_eye(frame, lm, RIGHT_EYE, w, h)

        eyes_batch = []
        for eye in [left_eye, right_eye]:
            if eye is not None:
                eyes_batch.append(eye)

        probs = []
        if eyes_batch:
            batch = np.concatenate(eyes_batch, axis=0)
            preds = model.predict(batch, verbose=0).reshape(-1)
            probs = [float(p) for p in preds]

        eyes_visible = len(eyes_batch)
        if probs:
            eye_prob = float(sum(probs) / len(probs))

        left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear_values = [v for v in [left_ear, right_ear] if v is not None]
        if ear_values:
            mean_ear = float(sum(ear_values) / len(ear_values))

        # Require both eyes visible to trust drowsiness detection.
        if eyes_visible < 2:
            drowsy_valid = False

        if drowsy_valid:
            closed_prob_model = closed_prob_from_model(eye_prob)
            closed_prob_ear = closed_prob_from_ear(mean_ear)

            if closed_prob_ear is None:
                fused_closed_prob = closed_prob_model
            else:
                fused_closed_prob = (0.70 * closed_prob_model) + (0.30 * closed_prob_ear)

    return {
        "face_detected": face_detected,
        "eye_prob": eye_prob,
        "mean_ear": mean_ear,
        "closed_prob_model": closed_prob_model,
        "closed_prob_ear": closed_prob_ear,
        "fused_closed_prob": fused_closed_prob,
        "head_pose": head_pose,
        "drowsy_valid": drowsy_valid,
        "eyes_visible": eyes_visible if face_detected else 0,
    }


def _run_calibration_phase(cap, seconds, line1, line2):
    start_time = time.time()
    ear_samples = []
    closed_model_samples = []

    while True:
        elapsed = time.time() - start_time
        if elapsed >= seconds:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        metrics = extract_eye_metrics(frame)
        if metrics["face_detected"] and metrics["drowsy_valid"]:
            if metrics["mean_ear"] is not None:
                ear_samples.append(metrics["mean_ear"])
            closed_model_samples.append(metrics["closed_prob_model"])

        remaining = max(0, int(seconds - elapsed))
        cv2.putText(frame, "AUTO CALIBRATION", (10, 30), 0, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, line1, (10, 60), 0, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, line2, (10, 85), 0, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time left: {remaining}s", (10, 90), 0, 0.7, (255, 255, 255), 2)
        if metrics["face_detected"] and not metrics["drowsy_valid"]:
            cv2.putText(frame, "Keep head facing camera", (10, 115), 0, 0.7, (0, 0, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Calibration interrupted by ESC.")
            break

    return ear_samples, closed_model_samples


def run_auto_calibration(cap, seconds, close_seconds, mode="full"):
    global MODEL_CLOSED_THRESHOLD, EAR_CLOSED_THRESHOLD

    open_ear_samples = []
    open_closed_model_samples = []
    if mode in {"full", "open_only"} and seconds > 0:
        print(f"Starting phase-1 calibration for {seconds}s (open eyes baseline).")
        open_ear_samples, open_closed_model_samples = _run_calibration_phase(
            cap,
            seconds,
            "Phase 1/2: keep eyes open naturally",
            "Look at camera with normal blinking",
        )

    if mode == "close_only":
        # Use active thresholds as open baseline when collecting only phase-2.
        open_ear_samples = [EAR_CLOSED_THRESHOLD / 0.72]
        open_closed_model_samples = [max(0.0, min(1.0, MODEL_CLOSED_THRESHOLD - 0.18))]

    if not open_closed_model_samples:
        print("Calibration skipped: no open-baseline samples available.")
        return None

    closed_ear_samples = []
    closed_model_samples = []
    if mode in {"full", "close_only"} and close_seconds > 0:
        print(f"Starting phase-2 calibration for {close_seconds}s (blink/close baseline).")
        closed_ear_samples, closed_model_samples = _run_calibration_phase(
            cap,
            close_seconds,
            "Phase 2/2: do slow long blinks",
            "Close eyes for ~1s, then open",
        )

    model_open_median = float(np.median(open_closed_model_samples))
    if closed_model_samples:
        model_closed_median = float(np.median(closed_model_samples))
        proposed_model_threshold = float(np.clip((model_open_median + model_closed_median) / 2.0, 0.30, 0.80))
    else:
        proposed_model_threshold = float(np.clip(model_open_median + 0.18, 0.35, 0.75))

    proposed_ear_threshold = EAR_CLOSED_THRESHOLD
    if open_ear_samples and closed_ear_samples:
        ear_open_median = float(np.median(open_ear_samples))
        ear_closed_median = float(np.median(closed_ear_samples))
        proposed_ear_threshold = float(np.clip((ear_open_median + ear_closed_median) / 2.0, 0.13, 0.30))
    elif open_ear_samples:
        ear_open_median = float(np.median(open_ear_samples))
        proposed_ear_threshold = float(np.clip(ear_open_median * 0.72, 0.14, 0.28))

    MODEL_CLOSED_THRESHOLD = proposed_model_threshold
    EAR_CLOSED_THRESHOLD = proposed_ear_threshold

    print(
        "Calibration complete: "
        f"MODEL_CLOSED_THRESHOLD={MODEL_CLOSED_THRESHOLD:.3f}, "
        f"EAR_CLOSED_THRESHOLD={EAR_CLOSED_THRESHOLD:.3f}"
    )
    return {
        "model_closed_threshold": float(MODEL_CLOSED_THRESHOLD),
        "ear_closed_threshold": float(EAR_CLOSED_THRESHOLD),
        "open_samples": len(open_closed_model_samples),
        "closed_samples": len(closed_model_samples),
    }

# -------- CAMERA --------
cap = None
candidate_indices = [CAMERA_INDEX] + [i for i in range(5) if i != CAMERA_INDEX]

for idx in candidate_indices:
    if platform.system().lower() == "windows":
        candidate = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    else:
        candidate = cv2.VideoCapture(idx)

    if candidate.isOpened():
        cap = candidate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Camera opened at index {idx}")
        break

    candidate.release()

if cap is None:
    print("❌ Camera failed")
    print("Hint: another app may be using your webcam. Close other OpenCV/Camera apps and retry.")
    exit()

loaded_profile = load_driver_profile(DRIVER_PROFILE_NAME)
if loaded_profile:
    apply_driver_profile(loaded_profile)

if AUTO_CALIBRATE and (CALIBRATION_SECONDS > 0 or CALIBRATION_MODE == "close_only"):
    calibration_result = run_auto_calibration(cap, CALIBRATION_SECONDS, CALIBRATION_CLOSE_SECONDS, CALIBRATION_MODE)
    if calibration_result is not None:
        save_driver_profile(
            DRIVER_PROFILE_NAME,
            calibration_result["model_closed_threshold"],
            calibration_result["ear_closed_threshold"],
            metadata={
                "open_samples": calibration_result["open_samples"],
                "closed_samples": calibration_result["closed_samples"],
            },
        )

print("✅ FULL SYSTEM STARTED (ML + IMU)")
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# -------- MAIN LOOP --------
imu_data = None
last_sent_alert = ""
last_alert_tx_time = 0.0
medium_state_since = None
frame_idx = 0
cached_metrics = {
    "face_detected": False,
    "eye_prob": 0.0,
    "mean_ear": None,
    "closed_prob_model": 0.0,
    "closed_prob_ear": None,
    "fused_closed_prob": 0.0,
    "head_pose": None,
    "drowsy_valid": False,
    "eyes_visible": 0,
}

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape

    # ---- SERIAL READ ----
    msg = serial_reader.read_parsed()
    if msg:
        msg_type = msg.get("type")
        if msg_type == "imu":
            imu_data = msg.get("data")
        elif msg_type == "telemetry":
            if "student_count" in msg:
                student_count = msg["student_count"]
            if "distance_cm" in msg:
                last_distance = msg["distance_cm"]
        elif msg_type == "bpm":
            last_bpm = msg.get("bpm")
        elif msg_type == "event":
            last_event = msg.get("event", "")
        elif msg_type == "count":
            if "student_count" in msg:
                student_count = msg["student_count"]
            last_door = msg.get("door", last_door)
            last_event = msg.get("event", last_event)

    # ---- CAMERA PROCESS ----
    frame_idx += 1
    if frame_idx % INFERENCE_EVERY_N_FRAMES == 0:
        metrics = extract_eye_metrics(frame)
        cached_metrics = metrics
    else:
        metrics = cached_metrics
    eye_prob = metrics["eye_prob"]
    mean_ear = metrics["mean_ear"]
    fused_closed_prob = metrics["fused_closed_prob"]
    face_detected = metrics["face_detected"]
    drowsy_valid = metrics["drowsy_valid"]
    head_pose = metrics["head_pose"]
    eyes_visible = metrics.get("eyes_visible", 0)

    # ---- DROWSINESS ----
    score, perclos, duration = tracker.update(fused_closed_prob, face_detected=(face_detected and drowsy_valid))

    # ---- IMU ----
    imu_state = process_imu(imu_data)

    # ---- FUSION ----
    if imu_state == "fall":
        final_alert = "CRITICAL"
    elif score >= HIGH_ALERT_THRESHOLD and imu_state == "tilt":
        final_alert = "HIGH"
    elif score >= MEDIUM_ALERT_THRESHOLD:
        final_alert = "MEDIUM"
    else:
        final_alert = "NORMAL"

    # If face/pose is not valid, suppress vision-triggered alarms while driver checks road/mirrors.
    vision_valid = face_detected and drowsy_valid and eyes_visible >= 2
    if not vision_valid and final_alert in {"MEDIUM", "HIGH"}:
        final_alert = "NORMAL"

    key = -1

    # ---- SEND ALERT ----
    now = time.time()
    buzzer_level = final_alert
    if final_alert == "MEDIUM":
        if medium_state_since is None:
            medium_state_since = now
        medium_elapsed = now - medium_state_since
        if (not BUZZER_ON_MEDIUM) or (medium_elapsed < MEDIUM_BUZZER_DELAY_SEC):
            buzzer_level = "NORMAL"
    else:
        medium_state_since = None

    is_alert_state = buzzer_level in {"CRITICAL", "HIGH", "MEDIUM"}
    should_send = False

    if buzzer_level != last_sent_alert:
        should_send = True
    elif is_alert_state and (now - last_alert_tx_time) >= ALERT_RESEND_INTERVAL_SEC:
        # Keep sending ALERT periodically so Arduino latch stays active.
        should_send = True

    # Manual buzzer test shortcuts.
    if key == ord('b'):
        serial_writer.send("ALERT_TEST")
        print("Manual buzzer test: ALERT_TEST sent")
    elif key == ord('n'):
        serial_writer.send("OK")
        print("Manual buzzer test: OK sent")

    if should_send:
        serial_writer.send_alert_level(buzzer_level)
        last_sent_alert = buzzer_level
        last_alert_tx_time = now

    # ---- COLOR ----
    color = (0,255,0)
    if final_alert in ["CRITICAL", "HIGH"]:
        color = (0,0,255)
    elif final_alert == "MEDIUM":
        color = (0,255,255)

    # ---- DISPLAY ----
    cv2.putText(frame, f"EyeProb(raw): {eye_prob:.2f}", (10,30), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"ClosedProb: {fused_closed_prob:.2f}", (10,55), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"EAR: {mean_ear if mean_ear is not None else -1:.3f}", (260,55), 0, 0.6, (255,255,255), 2)
    if head_pose is not None:
        cv2.putText(frame, f"Yaw:{head_pose['yaw']:.2f} Pitch:{head_pose['pitch']:.2f} Roll:{head_pose['roll_deg']:.1f}", (10, 77), 0, 0.55, (220,220,220), 2)
        pose_text = "POSE:FRONTAL" if drowsy_valid else "POSE:TURNING"
        pose_color = (0, 255, 0) if drowsy_valid else (0, 165, 255)
        cv2.putText(frame, pose_text, (310, 77), 0, 0.55, pose_color, 2)
    else:
        cv2.putText(frame, "POSE:NO FACE", (310, 77), 0, 0.55, (0, 0, 255), 2)
    cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10,85), 0, 0.7, (255,255,255), 2)
    if final_alert == "MEDIUM" and medium_state_since is not None:
        cv2.putText(frame, f"MEDIUM delay: {max(0.0, MEDIUM_BUZZER_DELAY_SEC - (time.time() - medium_state_since)):.1f}s", (220,85), 0, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"IMU: {imu_state}", (10,110), 0, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Students: {student_count}", (10,140), 0, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"{last_event} @ {last_door}", (10,170), 0, 0.6, (200,200,255), 2)
    cv2.putText(frame, f"Dist: {last_distance}", (10,195), 0, 0.6, (255,255,200), 2)
    cv2.putText(frame, f"BPM: {last_bpm}", (220,195), 0, 0.6, (255,255,200), 2)

    cv2.putText(frame, f"FINAL: {final_alert}", (10,225), 0, 1.0, color, 3)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

# -------- CLEANUP --------
cap.release()
cv2.destroyAllWindows()
del landmarker