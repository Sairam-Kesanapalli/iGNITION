class FusionDecision:
    def decide(self, eye_score, imu_state):

        if imu_state == "fall":
            return "CRITICAL"

        if eye_score > 0.7 and imu_state == "tilt":
            return "HIGH"

        if eye_score > 0.7:
            return "MEDIUM"

        return "NORMAL"