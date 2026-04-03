import cv2
import os
import time

# -------- SETUP --------
BASE_DIR = "../../data/videos"
os.makedirs(os.path.join(BASE_DIR, "open"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "closed"), exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

# -------- STATE --------
current_label = "open"
recording = False
out = None

print("\nControls:")
print("o → OPEN eyes")
print("c → CLOSED eyes")
print("r → start/stop recording")
print("q → quit\n")

# -------- LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # show info
    cv2.putText(frame, f"Label: {current_label}", (10,30),
                0, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"Recording: {recording}", (10,60),
                0, 0.7, (0,0,255), 2)

    cv2.imshow("Smart Recorder", frame)

    # -------- RECORD --------
    if recording and out is not None:
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    # -------- CHANGE LABEL --------
    if key == ord('o'):
        current_label = "open"
        print("Switched to OPEN")

    elif key == ord('c'):
        current_label = "closed"
        print("Switched to CLOSED")

    # -------- START / STOP --------
    elif key == ord('r'):
        recording = not recording

        if recording:
            filename = f"{current_label}_{int(time.time()*1000)}.avi"
            path = os.path.join(BASE_DIR, current_label, filename)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(path, fourcc, 20.0,
                                  (frame.shape[1], frame.shape[0]))

            print(f"🎥 Recording started → {path}")

        else:
            if out:
                out.release()
                out = None
            print("⏹ Recording stopped")

    # -------- EXIT --------
    elif key == ord('q'):
        break

# -------- CLEANUP --------
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()