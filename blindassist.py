import cv2
import pyttsx3
import time
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import speech_recognition as sr
import smtplib
from email.mime.text import MIMEText

# ─── Load Model ─────────────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")

# ─── Speech Engine ──────────────────────────────────────────────────────────
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
speech_lock = threading.Lock()

def speak(text):
    def _speak():
        with speech_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ─── Email Config — UPDATE THESE ────────────────────────────────────────────
SENDER_EMAIL    = "preranaprakash129@gmail.com"
SENDER_PASSWORD = "pgob yhus mabf nbfu"
EMERGENCY_EMAIL = "bsugunarao@gmail.com"
USER_NAME       = "the user"

def send_emergency_alert():
    try:
        msg = MIMEText(
            f"EMERGENCY ALERT\n\n"
            f"{USER_NAME} has fallen and may need help.\n"
            f"This alert was sent automatically by BlindAssist.\n\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Please check on them immediately."
        )
        msg['Subject'] = "BlindAssist Emergency Alert - Fall Detected"
        msg['From']    = SENDER_EMAIL
        msg['To']      = EMERGENCY_EMAIL
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, EMERGENCY_EMAIL, msg.as_string())
        speak("Emergency alert sent to your contact.")
        add_log("Emergency alert sent.")
    except Exception as e:
        add_log(f"Alert failed: {e}")
        speak("Could not send alert. Please call for help.")

# ─── Speech Recognition ─────────────────────────────────────────────────────
recognizer = sr.Recognizer()

def listen_for_command(timeout=5):
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=4)
            command = recognizer.recognize_google(audio).lower()
            return command
    except:
        return ""

# ─── Fall Detection ──────────────────────────────────────────────────────────
fall_pending = False

def check_for_fall(box_data):
    global fall_pending
    if fall_pending:
        return
    for box in box_data:
        if box['label'] != 'person':
            continue
        x1, y1, x2, y2 = box['coords']
        height = y2 - y1
        width  = x2 - x1
        if width > height * 1.3:
            fall_pending = True
            trigger_fall_alert()
            return

def trigger_fall_alert():
    speak("Warning. Fall detected. Sending emergency alert in 10 seconds. Say cancel to stop.")
    add_log("Fall detected! Say cancel within 10 seconds to stop alert.")
    status_icon.config(text="SOS")
    status_text.config(text="Fall Detected!\nSay CANCEL to stop", fg="#FF4444")

    def countdown():
        global fall_pending
        command = listen_for_command(timeout=9)
        if "cancel" in command:
            fall_pending = False
            speak("Alert cancelled.")
            add_log("Alert cancelled by user.")
            status_icon.config(text="OK")
            status_text.config(text="Path is clear", fg="#00FF94")
        else:
            add_log("No cancel received. Sending alert...")
            send_emergency_alert()
            fall_pending = False

    threading.Thread(target=countdown, daemon=True).start()

# ─── UI ──────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("BlindAssist Navigation")
root.configure(bg="#0a0a0a")
root.geometry("900x660")
root.resizable(False, False)

title_font  = font.Font(family="Courier New", size=22, weight="bold")
status_font = font.Font(family="Courier New", size=13)
label_font  = font.Font(family="Courier New", size=10)
badge_font  = font.Font(family="Courier New", size=11, weight="bold")

header = tk.Frame(root, bg="#0a0a0a", pady=10)
header.pack(fill="x", padx=20)
tk.Label(header, text="BLINDASSIST", font=title_font,
         bg="#0a0a0a", fg="#00FF94").pack(side="left")
tk.Label(header, text="NAVIGATION SYSTEM  v2.0", font=label_font,
         bg="#0a0a0a", fg="#444444").pack(side="left", padx=16, pady=6)
live_dot = tk.Label(header, text="STANDBY", font=badge_font,
                    bg="#0a0a0a", fg="#888888")
live_dot.pack(side="right")

content = tk.Frame(root, bg="#0a0a0a")
content.pack(fill="both", expand=True, padx=20)

cam_border = tk.Frame(content, bg="#333333", padx=2, pady=2)
cam_border.pack(side="left")
cam_label = tk.Label(cam_border, bg="#0a0a0a", width=620, height=460,
                     text="Say  'START'  to begin navigation",
                     font=font.Font(family="Courier New", size=16), fg="#444444")
cam_label.pack()

right = tk.Frame(content, bg="#0a0a0a", width=260, padx=16)
right.pack(side="right", fill="y")
right.pack_propagate(False)

tk.Label(right, text="STATUS", font=label_font,
         bg="#0a0a0a", fg="#444444").pack(anchor="w", pady=(10, 4))
status_box = tk.Frame(right, bg="#111111", padx=12, pady=12)
status_box.pack(fill="x")
status_icon = tk.Label(status_box, text="MIC",
                        font=font.Font(family="Courier New", size=20, weight="bold"),
                        bg="#111111", fg="#888888")
status_icon.pack()
status_text = tk.Label(status_box, text="Waiting for voice\ncommand...",
                        font=status_font, bg="#111111",
                        fg="#888888", wraplength=200, justify="center")
status_text.pack(pady=(6, 0))

tk.Frame(right, bg="#222222", height=1).pack(fill="x", pady=14)
tk.Label(right, text="DETECTION LOG", font=label_font,
         bg="#0a0a0a", fg="#444444").pack(anchor="w", pady=(0, 6))
log_frame = tk.Frame(right, bg="#111111", padx=10, pady=10)
log_frame.pack(fill="both", expand=True)
log_text = tk.Text(log_frame, bg="#111111", fg="#888888",
                   font=font.Font(family="Courier New", size=9),
                   relief="flat", state="disabled", wrap="word", width=24, height=12)
log_text.pack(fill="both", expand=True)

tk.Label(right, text='Say "start" or "stop"',
         font=label_font, bg="#0a0a0a", fg="#444444").pack(pady=(8, 0))
tk.Button(right, text="STOP APP", font=badge_font,
          bg="#FF4444", fg="white", relief="flat",
          padx=10, pady=6, cursor="hand2",
          command=root.destroy).pack(fill="x", pady=(8, 10))

footer = tk.Frame(root, bg="#0a0a0a", pady=6)
footer.pack(fill="x", padx=20)
tk.Label(footer, text="Smartphone Mode  |  Offline Voice  |  YOLOv8  |  Fall Detection",
         font=label_font, bg="#0a0a0a", fg="#333333").pack(side="left")
fps_label = tk.Label(footer, text="FPS: --", font=label_font,
                     bg="#0a0a0a", fg="#333333")
fps_label.pack(side="right")

# ─── State ───────────────────────────────────────────────────────────────────
is_running    = False
last_detected = []
frame_count   = 0
fps_time      = time.time()
cap           = cv2.VideoCapture(0)  # Change to 1 for iPhone via DroidCam

def add_log(message):
    timestamp = time.strftime("%H:%M:%S")
    log_text.config(state="normal")
    log_text.insert("end", f"[{timestamp}] {message}\n")
    log_text.see("end")
    log_text.config(state="disabled")

# ─── Voice Listener Thread ───────────────────────────────────────────────────
def voice_listener():
    global is_running
    speak("BlindAssist ready. Say start to begin navigation.")
    add_log("Listening for voice commands...")
    while True:
        command = listen_for_command(timeout=10)
       if "letsgo wayfr" in command and not is_running:
            is_running = True
            live_dot.config(text="LIVE", fg="#FF4444")
            cam_border.config(bg="#00FF94")
            status_icon.config(text="OK", fg="#00FF94")
            status_text.config(text="Path is clear", fg="#00FF94")
            speak("Navigation started. I will alert you of obstacles ahead.")
            add_log("Navigation started.")
        elif "stop wayfr" in command and is_running:
            is_running = False
            live_dot.config(text="STANDBY", fg="#888888")
            cam_border.config(bg="#333333")
            status_icon.config(text="MIC", fg="#888888")
            status_text.config(text="Waiting for voice\ncommand...", fg="#888888")
            speak("Navigation paused. Say letsgo wayfr to resume.")
            add_log("Navigation paused.")

threading.Thread(target=voice_listener, daemon=True).start()

# ─── Main Loop ───────────────────────────────────────────────────────────────
def update():
    global last_detected, frame_count, fps_time

    ret, frame = cap.read()
    if not ret:
        root.after(30, update)
        return

    frame_count += 1
    if time.time() - fps_time >= 1.0:
        fps_label.config(text=f"FPS: {frame_count}")
        frame_count = 0
        fps_time = time.time()

    if is_running:
        results     = model(frame, verbose=False)
        frame_width = frame.shape[1]
        detected    = []
        box_data    = []

        for result in results:
            for box in result.boxes:
                label      = model.names[int(box.cls)]
                confidence = float(box.conf)
                if confidence > 0.7:
                    x1, y1, x2, y2 = box.xyxy[0]
                    box_width    = x2 - x1
                    box_center_x = (x1 + x2) / 2
                    direction = (
                        "on your left"  if box_center_x < frame_width / 3 else
                        "on your right" if box_center_x > 2 * frame_width / 3 else
                        "in the centre"
                    )
                    distance = (
                        "immediately ahead" if box_width > frame_width * 0.4 else
                        "nearby"            if box_width > frame_width * 0.2 else
                        "in the distance"
                    )
                    detected.append((distance, f"{label} {distance} {direction}"))
                    box_data.append({
                        'label':  label,
                        'coords': (float(x1), float(y1), float(x2), float(y2))
                    })

        if not fall_pending:
            check_for_fall(box_data)

        detected.sort(key=lambda x: (
            0 if "immediately" in x[0] else
            1 if "nearby"      in x[0] else 2
        ))
        unique = sorted(set([d[1] for d in detected]))

        if unique and unique != last_detected:
            top     = [d[1] for d in detected[:2]]
            message = "Warning. " + ". ".join(top)
            speak(message)
            add_log(message)
            status_icon.config(text="!!", fg="#FF4444")
            status_text.config(text="\n".join(top), fg="#FF4444")
            last_detected = unique
        elif not detected and last_detected:
            speak("Path is clear")
            add_log("Path is clear")
            status_icon.config(text="OK", fg="#00FF94")
            status_text.config(text="Path is clear", fg="#00FF94")
            last_detected = []

        annotated     = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img           = Image.fromarray(annotated_rgb)
        img           = img.resize((620, 460))
        imgtk         = ImageTk.PhotoImage(image=img)
        cam_label.imgtk = imgtk
        cam_label.config(image=imgtk, text="")

    root.after(30, update)

# ─── Launch ──────────────────────────────────────────────────────────────────
add_log("BlindAssist v2.0 started.")
add_log("Say 'start' to begin.")
root.after(100, update)
root.mainloop()
cap.release()
