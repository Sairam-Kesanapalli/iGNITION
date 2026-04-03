import serial


class BusSerialReader:
	"""Read and parse serial messages from Arduino/IMU sources."""

	def __init__(self, port, baud=115200, timeout=1.0):
		self.port = port
		self.baud = baud
		self.timeout = timeout
		self.ser = None

	def connect(self):
		try:
			self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
			return True
		except Exception:
			self.ser = None
			return False

	def connected(self):
		return self.ser is not None and self.ser.is_open

	def read_parsed(self):
		if not self.connected():
			return None

		try:
			line = self.ser.readline().decode(errors="ignore").strip()
		except Exception:
			return None

		if not line:
			return None

		return parse_serial_line(line)


def _parse_kv_pairs(line, pair_sep="|", kv_sep=":"):
	result = {}
	parts = line.split(pair_sep)
	for part in parts:
		if kv_sep not in part:
			continue
		key, value = part.split(kv_sep, 1)
		result[key.strip().upper()] = value.strip()
	return result


def parse_serial_line(line):
	"""
	Supported incoming formats:
	- DIST:120|COUNT:3
	- BPM:72
	- ENTER detected / EXIT detected
	- COUNT:3,DOOR:FRONT,EVENT:ENTER
	- ax,ay,az,gx,gy,gz
	"""
	line = line.strip()
	if not line:
		return None

	upper_line = line.upper()

	if upper_line.startswith("DIST:"):
		fields = _parse_kv_pairs(line, pair_sep="|", kv_sep=":")
		msg = {"type": "telemetry", "raw": line}
		if "DIST" in fields:
			try:
				msg["distance_cm"] = float(fields["DIST"])
			except ValueError:
				pass
		if "COUNT" in fields:
			try:
				msg["student_count"] = int(fields["COUNT"])
			except ValueError:
				pass
		return msg

	if upper_line.startswith("BPM:"):
		try:
			bpm = int(line.split(":", 1)[1].strip())
			return {"type": "bpm", "bpm": bpm, "raw": line}
		except Exception:
			return None

	if upper_line == "ENTER DETECTED":
		return {"type": "event", "event": "ENTER", "raw": line}

	if upper_line == "EXIT DETECTED":
		return {"type": "event", "event": "EXIT", "raw": line}

	if upper_line.startswith("COUNT:") and "," in line:
		fields = _parse_kv_pairs(line, pair_sep=",", kv_sep=":")
		msg = {"type": "count", "raw": line}
		if "COUNT" in fields:
			try:
				msg["student_count"] = int(fields["COUNT"])
			except ValueError:
				pass
		if "DOOR" in fields:
			msg["door"] = fields["DOOR"]
		if "EVENT" in fields:
			msg["event"] = fields["EVENT"]
		return msg

	# Raw IMU CSV: ax,ay,az,gx,gy,gz
	if "," in line:
		parts = [p.strip() for p in line.split(",")]
		if len(parts) == 6:
			try:
				ax, ay, az, gx, gy, gz = map(float, parts)
				return {
					"type": "imu",
					"data": (ax, ay, az, gx, gy, gz),
					"raw": line,
				}
			except ValueError:
				return None

	return None
