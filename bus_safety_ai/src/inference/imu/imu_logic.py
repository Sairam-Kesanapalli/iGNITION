import math

class IMUProcessor:
    def __init__(self):
        self.last_fall_time = 0

    def process(self, data):
        if data is None:
            return "unknown"

        ax, ay, az, gx, gy, gz = data

        # -------- HEAD TILT --------
        pitch = math.degrees(math.atan2(ax, az))

        if abs(pitch) > 35:
            return "tilt"

        # -------- FALL DETECTION --------
        acc_mag = (ax**2 + ay**2 + az**2) ** 0.5

        if acc_mag > 3:
            return "fall"

        return "normal"