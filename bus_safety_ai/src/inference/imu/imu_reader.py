import serial

class IMUReader:
    def __init__(self, port="COM3", baud=115200):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            print("✅ IMU Connected")
        except:
            self.ser = None
            print("❌ IMU Not Connected")

    def read(self):
        if self.ser is None:
            return None

        try:
            line = self.ser.readline().decode().strip()
            ax, ay, az, gx, gy, gz = map(float, line.split(","))
            return ax, ay, az, gx, gy, gz
        except:
            return None