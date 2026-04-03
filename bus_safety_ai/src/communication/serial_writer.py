import os


class BusSerialWriter:
	"""Send control messages to a connected serial device."""

	def __init__(self, serial_instance):
		self.ser = serial_instance

	def send(self, message):
		if self.ser is None:
			return False

		try:
			self.ser.write((message + "\n").encode("utf-8"))
			return True
		except Exception:
			return False

	def send_alert_level(self, level):
		"""
		Maps high-level alert state to Arduino command.
		Arduino sketch currently reacts to any ALERT* command.
		"""
		alert_on_medium = os.getenv("ALERT_ON_MEDIUM", "1").strip().lower() in {"1", "true", "yes", "on"}
		allowed_levels = {"CRITICAL", "HIGH"}
		if alert_on_medium:
			allowed_levels.add("MEDIUM")

		if level in allowed_levels:
			return self.send(f"ALERT_{level.replace(' ', '_')}")
		return self.send("OK")
