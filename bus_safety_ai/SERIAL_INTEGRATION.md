# Serial Integration: Arduino <-> ML System

## 1) Flash Arduino sketch

Use the sketch in:
- ../sketch_feb02a/sketch_feb02a.ino

The sketch now:
- sends telemetry lines like `DIST:123|COUNT:4`
- sends structured events like `COUNT:4,DOOR:MAIN,EVENT:ENTER`
- accepts alert commands like `ALERT_CRITICAL`, `ALERT_HIGH`

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
BUS_SERIAL_PORT=/dev/ttyUSB0 BUS_SERIAL_BAUD=115200 python src/system/main_system.py
```

Optional environment variables:
- `BUS_CAMERA_INDEX` (default: `0`)
- `FACE_LANDMARKER_PATH` (path to `face_landmarker.task`)
- `EYE_MODEL_PATH` (path to `eye_model.keras`)

Example:

```bash
BUS_SERIAL_PORT=/dev/ttyACM0 BUS_CAMERA_INDEX=1 python src/system/main_system.py
```

## 5) Important serial note

Only one process can own the serial port at a time.

If Arduino Serial Monitor is open, close it before running Python.

## 6) Data flow summary

- Arduino -> Python: distance, count, events, BPM, optional IMU CSV
- Python -> Arduino: alert command lines (`ALERT_*` or `OK`)
