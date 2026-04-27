# ===================== SERIAL (ARDUINO) SETUP =====================
import serial
import time

# CHANGE COM PORT AS PER YOUR SYSTEM (e.g. COM3, COM4, COM5...)
COM_PORT = 'COM3'

try:
    arduino = serial.Serial(COM_PORT, 9600)
    time.sleep(2)  # Allow Arduino to reset
    print("[OK] Arduino connected on", COM_PORT)
except serial.SerialException as e:
    arduino = None
    print(f"[WARNING] Arduino NOT connected ({e})")
    print("   -> Fall detection will run WITHOUT Arduino alerts.")
    print("   -> Check your COM port in Device Manager and update COM_PORT in this script.")

# ===================== IMPORT LIBRARIES =====================
import cv2
import math
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os

print("OpenCV version:", cv2.__version__)

# ===================== CONFIGURATION =====================
VIDEO_SOURCE = 0              # 0 = Webcam | or "video.mp4"
MODEL_NAME = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5
FALL_ANGLE_THRESHOLD = 45     # degrees
FALL_TIME_THRESHOLD = 20      # frames

# ===================== UTILITY FUNCTIONS =====================
def calculate_angle(p1, p2):
    """Calculate angle between two points"""
    x1, y1 = p1
    x2, y2 = p2
    dy = abs(y2 - y1)
    dx = abs(x2 - x1)
    return math.degrees(math.atan2(dy, dx))

def send_alert(person_id):
    """Send alert to Arduino"""
    print(f"[ALERT] FALL DETECTED for Person ID: {person_id}")
    if arduino:
        arduino.write(b'1')   # Turn ON alert (LED/Buzzer)

# ===================== MAIN FUNCTION =====================
def main():
    print("Loading YOLOv8 Pose Model...")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("[ERROR] Could not open video source")
        return

    track_history = defaultdict(int)
    alert_sent_registry = defaultdict(bool)

    print("[RUNNING] Fall Detection System Running (Press Q to quit)")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy()

            for box, track_id, kp in zip(boxes, track_ids, keypoints):

                # Keypoint confidence check (Shoulders + Hips)
                if (kp[5][2] < CONFIDENCE_THRESHOLD or kp[6][2] < CONFIDENCE_THRESHOLD or
                    kp[11][2] < CONFIDENCE_THRESHOLD or kp[12][2] < CONFIDENCE_THRESHOLD):
                    continue

                id_int = int(track_id)
                x1, y1, x2, y2 = map(int, box)

                # Midpoints
                shoulder_mid = (
                    int((kp[5][0] + kp[6][0]) / 2),
                    int((kp[5][1] + kp[6][1]) / 2)
                )
                hip_mid = (
                    int((kp[11][0] + kp[12][0]) / 2),
                    int((kp[11][1] + kp[12][1]) / 2)
                )

                angle = calculate_angle(hip_mid, shoulder_mid)

                state_color = (0, 255, 0)
                status_text = f"Normal ({int(angle)}°)"

                # ================= FALL LOGIC =================
                if angle < FALL_ANGLE_THRESHOLD:
                    track_history[id_int] += 1
                    state_color = (0, 255, 255)
                    status_text = f"Unstable ({track_history[id_int]}/{FALL_TIME_THRESHOLD})"

                    if track_history[id_int] > FALL_TIME_THRESHOLD:
                        state_color = (0, 0, 255)
                        status_text = "FALL DETECTED"

                        if not alert_sent_registry[id_int]:
                            send_alert(id_int)
                            alert_sent_registry[id_int] = True
                else:
                    track_history[id_int] = 0
                    alert_sent_registry[id_int] = False
                    if arduino:
                        arduino.write(b'0')  # Turn OFF alert

                # ================= VISUALIZATION =================
                cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2)
                cv2.line(frame, shoulder_mid, hip_mid, state_color, 3)
                cv2.putText(
                    frame,
                    f"ID {id_int}: {status_text}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    state_color,
                    2
                )

        cv2.imshow("Fall Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.write(b'0')
        arduino.close()
    print("[STOPPED] System stopped")

# ===================== RUN =====================
if __name__ == "__main__":
    main()