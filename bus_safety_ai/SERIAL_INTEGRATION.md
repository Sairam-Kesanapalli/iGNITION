# Serial Integration: Arduino <-> ML System

## 1) Flash Arduino sketch

Use the sketch in:
- ../sketch_feb02a/sketch_feb02a.ino

The sketch now:
- sends telemetry lines like `DIST:123|COUNT:4`
- sends structured events like `COUNT:4,DOOR:MAIN,EVENT:ENTER`
- accepts alert commands like `ALERT_CRITICAL`, `ALERT_HIGH`
- supports direct test commands: `BUZZER_TEST`, `BUZZER_ON`, `BUZZER_PULSE`

## 2) Install Python dependencies

From bus_safety_ai:

```bash
pip install -r requirements.txt
```

## 3) Find your serial port on Linux

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

Use the detected port in `BUS_SERIAL_PORT`.

## 4) Run the ML system with serial bridge

From bus_safety_ai:

```bash
BUS_SERIAL_PORT=/dev/ttyACM0 BUS_SERIAL_BAUD=74880 python src/system/main_system.py
```

Optional environment variables:
- `BUS_SERIAL_PORT` (default: `auto`, auto-detects available serial port)
- `BUS_SERIAL_BAUD` (default: `74880`)
- `BUS_CAMERA_INDEX` (default: `0`)
- `FACE_LANDMARKER_PATH` (path to `face_landmarker.task`)
- `EYE_MODEL_PATH` (path to `eye_model.keras`)
- `HEAD_MODEL_PATH` (path to `head_model.keras`)
- `HEAD_SCALER_PATH` (path to `head_scaler.npz`)
- `HEAD_CLASSES_PATH` (path to `head_classes.json`)
- `ALERT_ON_MEDIUM` (default: `1`)
- `MEDIUM_BUZZER_DELAY_SEC` (default: `1.0`)

Example:

```bash
AUTO_CALIBRATE=0 DRIVER_PROFILE=sairam BUS_SERIAL_PORT=/dev/ttyACM0 python src/system/main_system.py
```

## 5) Head model setup

Train and export head artifacts:

```bash
python src/head/train_head_model.py
```

Expected output files:
- `models/head_model.keras`
- `models/head_scaler.npz`
- `models/head_classes.json`

## 6) Head-only calibration (keeps eye limits unchanged)

```bash
AUTO_CALIBRATE=1 CALIBRATION_MODE=head_only CALIBRATION_SECONDS=20 HEAD_AUTO_CALIBRATE=1 DRIVER_PROFILE=sairam python src/system/main_system.py
```

This updates only head baseline/threshold fields in the driver profile.

## 7) Important serial note

Only one process can own the serial port at a time.

If Arduino Serial Monitor is open, close it before running Python.

## 8) Data flow summary

- Arduino -> Python: distance, count, events, BPM, optional IMU CSV
- Python -> Arduino: alert command lines (`ALERT_*` or `OK`)

## 9) Runtime quick test keys

In the OpenCV window:
- `b`: send high alert (`ALERT_HIGH`) to test buzzer
- `n`: send normal/clear (`OK`) to stop buzzer
