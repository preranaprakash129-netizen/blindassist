import cv2
import pyttsx3
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize pyttsx3 for offline speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)  # Max volume

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Open camera (0 = laptop, 1 = iPhone via DroidCam)
cap = cv2.VideoCapture(0)
last_detected = []

print("BlindAssist is running... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    frame_width = frame.shape[1]
    detected = []

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            confidence = float(box.conf)

            if confidence > 0.7:
                x1, y1, x2, y2 = box.xyxy[0]
                box_width = x2 - x1
                box_center_x = (x1 + x2) / 2

                # Direction
                if box_center_x < frame_width / 3:
                    direction = "on your left"
                elif box_center_x > 2 * frame_width / 3:
                    direction = "on your right"
                else:
                    direction = "in the centre"

                # Distance
                if box_width > frame_width * 0.4:
                    distance = "immediately ahead"
                elif box_width > frame_width * 0.2:
                    distance = "nearby"
                else:
                    distance = "in the distance"

                detected.append((distance, f"{label} {distance} {direction}"))

    # Sort by closest first
    detected.sort(key=lambda x: (
        0 if "immediately" in x[0] else 1 if "nearby" in x[0] else 2
    ))

    unique = sorted(set([d[1] for d in detected]))
    if unique and unique != last_detected:
        # Only announce closest 2 obstacles max
        top = [d[1] for d in detected[:2]]
        message = "Warning. " + ". ".join(top)
        print(message)
        speak(message)
        last_detected = unique
    elif not detected and last_detected:
        speak("Path is clear")
        last_detected = []

    cv2.imshow("BlindAssist Navigation", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
