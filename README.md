# DriveSafe 360

Real-time intelligent school bus safety system built with ESP32/Arduino sensors and a PC-based AI server.

## Project Summary

### Problem Statement
Design a real-time intelligent school bus safety system that monitors driver condition and passenger status, and generates instant alerts to prevent accidents and improve child safety.

### Solution Overview
DriveSafe 360 combines:
- Embedded sensing and actuation on ESP32/Arduino
- Webcam-based AI inference on PC
- Two-way serial communication between Arduino and AI server

The system continuously monitors driver drowsiness/head behavior and bus environment signals, then triggers audible alerts when unsafe conditions are detected.

## Current Features (Implemented)

- Driver drowsiness detection from webcam (eye model + temporal logic)
- Head movement state detection (trained head model + fallback logic)
- Smart passenger counting using ultrasonic trend detection
- Real-time buzzer alerting from AI server to Arduino
- Safety latching for environment alerts (continues until last child exits)
- Optional MQ7 alcohol sensing integration
- Optional DHT11 temperature sensing integration
- Driver profile support (including head-only calibration mode)

## Working Principle

1. Arduino reads sensors (ultrasonic, optional MQ7, optional DHT11).
2. PC AI server processes webcam + serial sensor stream in real time.
3. Fusion logic classifies risk level.
4. Alerts are sent back to Arduino (`ALERT_*` or `OK`) over serial.
5. Buzzer and LCD indicate active risk and bus state.

## Applications

- School buses
- Public transport
- Fleet and driver monitoring systems

## Future Scope

- Heart-rate monitoring integration
- SMS alert delivery

## Team

- K. Sairam (Lead) - Roll No: 24BEC019
- J. Durgesh - Roll No: 24BEC016
- R. Mohan - Roll No: 24BDS066
- P. Avinash - Roll No: 24BEC036

## Contributions

- K. Sairam: ESP/Arduino coding and system integration
- R. Mohan: ML implementation
- J. Durgesh and P. Avinash: wiring, ML-Arduino interface integration, and presentation/slides

## Repository Note

Commit history may show only Sairam in some phases. This was intentional to reduce merge conflicts during rapid integration. Team contributions are shared across implementation, hardware, and integration work.

## Quick Setup

### 1) Python side

From `bus_safety_ai/`:

```bash
pip install -r requirements.txt
```

Run:

```bash
AUTO_CALIBRATE=0 DRIVER_PROFILE=sairam BUS_SERIAL_PORT=/dev/ttyACM0 BUS_SERIAL_BAUD=74880 python src/system/main_system.py
```

### 2) Arduino side

Flash:

```text
sketch_feb02a/sketch_feb02a.ino
```

### 3) Optional sensor modes

In the sketch:
- `MQ7_ENABLED = true/false` to enable/disable alcohol sensor logic
- When disabled, LCD shows distance in MQ display area
