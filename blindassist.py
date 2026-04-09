import cv2
import pyttsx3
import time
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize pyttsx3 for offline speech
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

# ─── UI Setup ───────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("BlindAssist Navigation")
root.configure(bg="#0a0a0a")
root.geometry("900x620")
root.resizable(False, False)

# Fonts
title_font   = font.Font(family="Courier New", size=22, weight="bold")
status_font  = font.Font(family="Courier New", size=13)
label_font   = font.Font(family="Courier New", size=10)
badge_font   = font.Font(family="Courier New", size=11, weight="bold")

# ─── Header ─────────────────────────────────────────────────────────────────
header = tk.Frame(root, bg="#0a0a0a", pady=10)
header.pack(fill="x", padx=20)

tk.Label(header, text="👁  BLINDASSIST", font=title_font,
         bg="#0a0a0a", fg="#00FF94").pack(side="left")

tk.Label(header, text="NAVIGATION SYSTEM  v1.0", font=label_font,
         bg="#0a0a0a", fg="#444444").pack(side="left", padx=16, pady=6)

# Live indicator
live_dot = tk.Label(header, text="● LIVE", font=badge_font,
                    bg="#0a0a0a", fg="#FF4444")
live_dot.pack(side="right")

# ─── Main content area ──────────────────────────────────────────────────────
content = tk.Frame(root, bg="#0a0a0a")
content.pack(fill="both", expand=True, padx=20)

# Camera feed (left)
cam_border = tk.Frame(content, bg="#00FF94", padx=2, pady=2)
cam_border.pack(side="left")

cam_label = tk.Label(cam_border, bg="#0a0a0a")
cam_label.pack()

# Right panel
right = tk.Frame(content, bg="#0a0a0a", width=260, padx=16)
right.pack(side="right", fill="y")
right.pack_propagate(False)

# Status box
tk.Label(right, text="STATUS", font=label_font,
         bg="#0a0a0a", fg="#444444").pack(anchor="w", pady=(10, 4))

status_box = tk.Frame(right, bg="#111111", padx=12, pady=12)
status_box.pack(fill="x")

status_icon = tk.Label(status_box, text="✅", font=font.Font(size=24),
                        bg="#111111")
status_icon.pack()

status_text = tk.Label(status_box, text="Path is clear",
                        font=status_font, bg="#111111",
                        fg="#00FF94", wraplength=200, justify="center")
status_text.pack(pady=(6, 0))

# Divider
tk.Frame(right, bg="#222222", height=1).pack(fill="x", pady=14)

# Detection log
tk.Label(right, text="DETECTION LOG", font=label_font,
         bg="#0a0a0a", fg="#444444").pack(anchor="w", pady=(0, 6))

log_frame = tk.Frame(right, bg="#111111", padx=10, pady=10)
log_frame.pack(fill="both", expand=True)

log_text = tk.Text(log_frame, bg="#111111", fg="#888888",
                   font=font.Font(family="Courier New", size=9),
                   relief="flat", state="disabled", wrap="word",
                   width=24, height=12)
log_text.pack(fill="both", expand=True)

# Quit button
quit_btn = tk.Button(right, text="■  STOP", font=badge_font,
                     bg="#FF4444", fg="white", relief="flat",
                     padx=10, pady=6, cursor="hand2",
                     command=root.destroy)
quit_btn.pack(fill="x", pady=(14, 10))

# ─── Footer ─────────────────────────────────────────────────────────────────
footer = tk.Frame(root, bg="#0a0a0a", pady=6)
footer.pack(fill="x", padx=20)

tk.Label(footer, text="📍 Smartphone Mode  |  🔊 Offline Voice  |  🤖 YOLOv8",
         font=label_font, bg="#0a0a0a", fg="#333333").pack(side="left")

fps_label = tk.Label(footer, text="FPS: --", font=label_font,
                     bg="#0a0a0a", fg="#333333")
fps_label.pack(side="right")

# ─── Logic ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)   # Change to 1 for iPhone via DroidCam
last_detected = []
frame_count = 0
fps_time = time.time()

def add_log(message):
    timestamp = time.strftime("%H:%M:%S")
    log_text.config(state="normal")
    log_text.insert("end", f"[{timestamp}] {message}\n")
    log_text.see("end")
    log_text.config(state="disabled")

def update():
    global last_detected, frame_count, fps_time

    ret, frame = cap.read()
    if not ret:
        root.after(30, update)
        return

    # FPS counter
    frame_count += 1
    if time.time() - fps_time >= 1.0:
        fps_label.config(text=f"FPS: {frame_count}")
        frame_count = 0
        fps_time = time.time()

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
        top = [d[1] for d in detected[:2]]
        message = "Warning. " + ". ".join(top)
        speak(message)
        add_log(message)

        # Update status UI
        status_icon.config(text="⚠️")
        status_text.config(
            text="\n".join(top),
            fg="#FF4444"
        )
        last_detected = unique

    elif not detected and last_detected:
        speak("Path is clear")
        add_log("Path is clear")
        status_icon.config(text="✅")
        status_text.config(text="Path is clear", fg="#00FF94")
        last_detected = []

    # Show camera feed in UI
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(annotated_rgb)
    img = img.resize((620, 460))
    imgtk = ImageTk.PhotoImage(image=img)
    cam_label.imgtk = imgtk
    cam_label.config(image=imgtk)

    root.after(30, update)

# ─── Start ───────────────────────────────────────────────────────────────────
add_log("BlindAssist started.")
add_log("Waiting for obstacles...")
root.after(100, update)
root.mainloop()
cap.release()
