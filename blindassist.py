import cv2
import pygame
import time
from gtts import gTTS
from ultralytics import YOLO
import os
import tempfile

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize pygame for audio
pygame.mixer.init()

def speak(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        temp_path = f.name
    tts.save(temp_path)
    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()
    time.sleep(0.2)
    try:
        os.unlink(temp_path)
    except:
        pass

# Open camera (0 = laptop camera)
cap = cv2.VideoCapture(0)
last_spoken = 0
last_detected = []

print("BlindAssist is running... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    detected = []
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            confidence = float(box.conf)
            if confidence > 0.7:
                detected.append(label)

    # Speak every 4 seconds
   # Only speak if new objects detected
    unique = sorted(set(detected))
    if unique and unique != last_detected:
        message = "Warning. " + " and ".join(unique) + " detected ahead."
        print(message)
        speak(message)
        last_detected = unique
        last_spoken = time.time()
    elif not detected and last_detected:
        last_detected = []
    cv2.imshow("BlindAssist Navigation", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()