import cv2
import pyttsx3
import time
import tkinter as tk
from tkinter import font, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import speech_recognition as sr

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

# ─── Global State ────────────────────────────────────────────────────────────
emergency_contact = {"name": "", "phone": ""}
is_running        = False
last_detected     = []
frame_count       = 0
fps_time          = time.time()
fall_pending      = False
cap               = None
root              = None

cam_label       = None
cam_border      = None
status_badge    = None
status_text     = None
status_icon_lbl = None
log_text        = None
fps_label       = None

# ─── Log Helper ──────────────────────────────────────────────────────────────
def add_log(message):
    if log_text is None:
        return
    timestamp = time.strftime("%H:%M")
    log_text.config(state="normal")
    log_text.insert("end", f"[{timestamp}] {message}\n")
    log_text.see("end")
    log_text.config(state="disabled")

# ─── Simulated SMS Alert Popup ───────────────────────────────────────────────
def simulate_sms_alert():
    name  = emergency_contact["name"]
    phone = emergency_contact["phone"]
    speak(f"Emergency alert sent to {name}.")
    add_log(f"SMS sent to {name} at {phone}.")

    popup = tk.Toplevel(root)
    popup.title("Alert Sent")
    popup.geometry("400x300")
    popup.configure(bg="white")
    popup.resizable(False, False)
    popup.grab_set()

    tk.Frame(popup, bg="#FF3B30", height=6).pack(fill="x")
    body = tk.Frame(popup, bg="white", padx=30, pady=24)
    body.pack(fill="both", expand=True)

    tk.Label(body, text="Emergency Alert Sent",
             font=font.Font(family="Helvetica", size=14, weight="bold"),
             bg="white", fg="#1c1c1e").pack(anchor="w")
    tk.Frame(body, bg="#e5e5ea", height=1).pack(fill="x", pady=12)
    tk.Label(body, text=f"Contact: {name}",
             font=font.Font(family="Helvetica", size=11),
             bg="white", fg="#1c1c1e").pack(anchor="w")
    tk.Label(body, text=f"Phone: {phone}",
             font=font.Font(family="Helvetica", size=11),
             bg="white", fg="#636366").pack(anchor="w", pady=(2, 0))
    tk.Label(body, text=f"Time: {time.strftime('%H:%M:%S')}",
             font=font.Font(family="Helvetica", size=11),
             bg="white", fg="#636366").pack(anchor="w", pady=(2, 12))
    tk.Label(body,
             text="Fall detected. Please check on the user immediately.",
             font=font.Font(family="Helvetica", size=10),
             bg="#f2f2f7", fg="#3a3a3c",
             wraplength=320, justify="left",
             padx=10, pady=8).pack(fill="x")
    tk.Button(body, text="Dismiss",
              font=font.Font(family="Helvetica", size=11, weight="bold"),
              bg="#007AFF", fg="white", relief="flat",
              padx=12, pady=6, cursor="hand2",
              command=popup.destroy).pack(pady=(14, 0))

# ─── Speech Recognition ──────────────────────────────────────────────────────
recognizer = sr.Recognizer()

def listen_for_command(timeout=5):
    try:
        with sr.Microphone(device_index=2) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=4)
            command = recognizer.recognize_google(audio).lower()
            return command
    except:
        return ""

# ─── Fall Detection ──────────────────────────────────────────────────────────
def trigger_fall_alert(event=None):
    global fall_pending
    if fall_pending:
        return
    fall_pending = True
    speak("Warning. Fall detected. Sending alert in 10 seconds. Say cancel to stop.")
    add_log("Fall detected! Say cancel within 10 seconds.")
    status_badge.config(text="FALL DETECTED", bg="#FF3B30", fg="white")
    status_text.config(text="Say CANCEL to stop alert", fg="#FF3B30")

    def countdown():
        global fall_pending
        command = listen_for_command(timeout=9)
        if "cancel" in command:
            fall_pending = False
            speak("Alert cancelled.")
            add_log("Alert cancelled by user.")
            status_badge.config(text="LIVE", bg="#34C759", fg="white")
            status_text.config(text="Path is clear", fg="#34C759")
        else:
            add_log("No cancel — sending alert...")
            root.after(0, simulate_sms_alert)
            fall_pending = False

    threading.Thread(target=countdown, daemon=True).start()

# ─── Voice Listener ──────────────────────────────────────────────────────────
def voice_listener():
    global is_running
    speak("BlindAssist ready. Say start navigation to begin.")
    add_log("Ready — listening for commands.")
    while True:
        command = listen_for_command(timeout=10)
        if "start navigation" in command and not is_running:
            is_running = True
            status_badge.config(text="LIVE", bg="#34C759", fg="white")
            cam_border.config(bg="#34C759")
            status_icon_lbl.config(text="ON")
            status_text.config(text="Scanning for obstacles...", fg="#34C759")
            speak("Navigation started.")
            add_log("Navigation started.")
        elif "stop navigation" in command and is_running:
            is_running = False
            status_badge.config(text="STANDBY", bg="#e5e5ea", fg="#636366")
            cam_border.config(bg="white")
            status_icon_lbl.config(text="MIC")
            status_text.config(text="Waiting for voice command", fg="#8e8e93")
            speak("Navigation paused.")
            add_log("Navigation paused.")

# ─── Camera Update Loop ──────────────────────────────────────────────────────
def update():
    global last_detected, frame_count, fps_time

    ret, frame = cap.read()
    if not ret:
        root.after(50, update)
        return

    frame_count += 1
    if time.time() - fps_time >= 1.0:
        fps_label.config(text=f"FPS: {frame_count}")
        frame_count = 0
        fps_time = time.time()

    if is_running:
        results     = model(frame, verbose=False, imgsz=320)
        frame_width = frame.shape[1]
        detected    = []

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
            status_icon_lbl.config(text="!!")
            status_text.config(text="\n".join(top), fg="#FF3B30")
            last_detected = unique
        elif not detected and last_detected:
            speak("Path is clear")
            add_log("Path is clear")
            status_icon_lbl.config(text="OK")
            status_text.config(text="Path is clear", fg="#34C759")
            last_detected = []

        annotated     = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img           = Image.fromarray(annotated_rgb)
        img           = img.resize((640, 480))
        imgtk         = ImageTk.PhotoImage(image=img)
        cam_label.imgtk = imgtk
        cam_label.config(image=imgtk, text="")

    root.after(50, update)

# ─── Onboarding Window ───────────────────────────────────────────────────────
def show_onboarding():
    win = tk.Tk()
    win.title("BlindAssist Setup")
    win.geometry("460x480")
    win.configure(bg="white")
    win.resizable(False, False)

    hdr = tk.Frame(win, bg="#007AFF", pady=24)
    hdr.pack(fill="x")
    tk.Label(hdr, text="BlindAssist",
             font=font.Font(family="Helvetica", size=24, weight="bold"),
             bg="#007AFF", fg="white").pack()
    tk.Label(hdr, text="Navigation for the visually impaired",
             font=font.Font(family="Helvetica", size=11),
             bg="#007AFF", fg="#cce4ff").pack(pady=(2, 0))

    body = tk.Frame(win, bg="white", padx=40, pady=28)
    body.pack(fill="both", expand=True)

    tk.Label(body, text="Emergency Contact Setup",
             font=font.Font(family="Helvetica", size=14, weight="bold"),
             bg="white", fg="#1c1c1e").pack(anchor="w")
    tk.Label(body, text="This person will be alerted if a fall is detected.",
             font=font.Font(family="Helvetica", size=10),
             bg="white", fg="#8e8e93").pack(anchor="w", pady=(2, 20))

    tk.Label(body, text="Contact Name",
             font=font.Font(family="Helvetica", size=10, weight="bold"),
             bg="white", fg="#3a3a3c").pack(anchor="w")
    name_var = tk.StringVar()
    tk.Entry(body, textvariable=name_var,
             font=font.Font(family="Helvetica", size=12),
             relief="flat", bg="#f2f2f7", fg="#1c1c1e",
             insertbackground="#007AFF").pack(fill="x", ipady=9, pady=(3, 14))

    tk.Label(body, text="Phone Number  (10 digits)",
             font=font.Font(family="Helvetica", size=10, weight="bold"),
             bg="white", fg="#3a3a3c").pack(anchor="w")
    phone_var = tk.StringVar()
    tk.Entry(body, textvariable=phone_var,
             font=font.Font(family="Helvetica", size=12),
             relief="flat", bg="#f2f2f7", fg="#1c1c1e",
             insertbackground="#007AFF").pack(fill="x", ipady=9, pady=(3, 24))

    def submit():
        name  = name_var.get().strip()
        phone = phone_var.get().strip()
        if not name or not phone or len(phone) != 10 or not phone.isdigit():
            messagebox.showerror("Invalid Input",
                                 "Please enter a valid name and 10-digit phone number.")
            return
        emergency_contact["name"]  = name
        emergency_contact["phone"] = phone
        win.destroy()
        launch_main()

    tk.Button(body, text="Start BlindAssist",
              font=font.Font(family="Helvetica", size=12, weight="bold"),
              bg="#007AFF", fg="white", relief="flat",
              padx=16, pady=10, cursor="hand2",
              command=submit).pack(fill="x")

    win.mainloop()

# ─── Main App Window ─────────────────────────────────────────────────────────
def launch_main():
    global root, cap
    global cam_label, cam_border
    global status_badge, status_text, status_icon_lbl
    global log_text, fps_label

    root = tk.Tk()
    root.title("BlindAssist")
    root.configure(bg="#f2f2f7")
    root.geometry("980x640")
    root.resizable(False, False)

    h1      = font.Font(family="Helvetica", size=18, weight="bold")
    body_f  = font.Font(family="Helvetica", size=11)
    small_f = font.Font(family="Helvetica", size=9)
    badge_f = font.Font(family="Helvetica", size=10, weight="bold")
    mono_f  = font.Font(family="Courier New", size=9)

    def stop_navigation():
        global is_running, last_detected, fall_pending
        is_running    = False
        fall_pending  = False
        last_detected = []
        status_badge.config(text="STANDBY", bg="#e5e5ea", fg="#636366")
        cam_border.config(bg="white")
        status_icon_lbl.config(text="MIC")
        status_text.config(text="Waiting for\nvoice command", fg="#8e8e93")
        speak("Navigation stopped.")
        add_log("Navigation stopped.")

    def start_navigation():
        global is_running
        is_running = True
        status_badge.config(text="LIVE", bg="#34C759", fg="white")
        cam_border.config(bg="#34C759")
        status_icon_lbl.config(text="ON")
        status_text.config(text="Scanning for\nobstacles...", fg="#34C759")
        speak("Navigation started.")
        add_log("Navigation started.")

    # Top bar
    topbar = tk.Frame(root, bg="white", pady=12, padx=24)
    topbar.pack(fill="x")

    tk.Label(topbar, text="BlindAssist", font=h1,
             bg="white", fg="#1c1c1e").pack(side="left")
    tk.Label(topbar, text="Navigation System  v2.0", font=body_f,
             bg="white", fg="#8e8e93").pack(side="left", padx=12)
    status_badge = tk.Label(topbar, text="STANDBY",
                             font=badge_f, bg="#e5e5ea",
                             fg="#636366", padx=12, pady=5)
    status_badge.pack(side="left", padx=8)

    tk.Button(topbar, text="Stop Navigation", font=badge_f,
              bg="#FF9500", fg="white", relief="flat",
              padx=12, pady=5, cursor="hand2",
              command=stop_navigation).pack(side="right", padx=8)

    tk.Button(topbar, text="Start Navigation", font=badge_f,
              bg="#34C759", fg="white", relief="flat",
              padx=12, pady=5, cursor="hand2",
              command=start_navigation).pack(side="right", padx=8)

    tk.Frame(root, bg="#e5e5ea", height=1).pack(fill="x")

    # Content area
    content = tk.Frame(root, bg="#f2f2f7", padx=20, pady=16)
    content.pack(fill="both", expand=True)

    # Camera panel
    cam_outer  = tk.Frame(content, bg="#e5e5ea", padx=1, pady=1)
    cam_outer.pack(side="left")
    cam_border = tk.Frame(cam_outer, bg="white", padx=3, pady=3)
    cam_border.pack()
    cam_label  = tk.Label(cam_border, bg="#1c1c1e", width=640, height=480,
                           text='Click Start Navigation to begin',
                           font=font.Font(family="Helvetica", size=14),
                           fg="#636366")
    cam_label.pack()

    # Right panel
    right = tk.Frame(content, bg="#f2f2f7", width=270, padx=16)
    right.pack(side="right", fill="y")
    right.pack_propagate(False)

    # Status card
    tk.Label(right, text="STATUS", font=small_f,
             bg="#f2f2f7", fg="#8e8e93").pack(anchor="w", pady=(0, 4))
    status_card = tk.Frame(right, bg="white", padx=16, pady=16,
                            highlightbackground="#e5e5ea", highlightthickness=1)
    status_card.pack(fill="x")
    status_icon_lbl = tk.Label(status_card, text="MIC",
                                font=font.Font(family="Helvetica", size=20, weight="bold"),
                                bg="white", fg="#8e8e93")
    status_icon_lbl.pack()
    status_text = tk.Label(status_card, text="Waiting for\nvoice command",
                            font=body_f, bg="white", fg="#8e8e93",
                            wraplength=210, justify="center")
    status_text.pack(pady=(6, 0))

    tk.Frame(right, bg="#e5e5ea", height=1).pack(fill="x", pady=14)

    # Emergency contact card
    tk.Label(right, text="EMERGENCY CONTACT", font=small_f,
             bg="#f2f2f7", fg="#8e8e93").pack(anchor="w", pady=(0, 4))
    contact_card = tk.Frame(right, bg="white", padx=16, pady=12,
                             highlightbackground="#e5e5ea", highlightthickness=1)
    contact_card.pack(fill="x")
    tk.Label(contact_card, text=f"Name: {emergency_contact['name']}",
             font=body_f, bg="white", fg="#1c1c1e").pack(anchor="w")
    tk.Label(contact_card, text=f"Phone: {emergency_contact['phone']}",
             font=body_f, bg="white", fg="#636366").pack(anchor="w", pady=(4, 0))

    tk.Frame(right, bg="#e5e5ea", height=1).pack(fill="x", pady=14)

    # Activity log
    tk.Label(right, text="ACTIVITY LOG", font=small_f,
             bg="#f2f2f7", fg="#8e8e93").pack(anchor="w", pady=(0, 4))
    log_card = tk.Frame(right, bg="white", padx=10, pady=10,
                         highlightbackground="#e5e5ea", highlightthickness=1)
    log_card.pack(fill="both", expand=True)
    log_text = tk.Text(log_card, bg="white", fg="#3a3a3c",
                        font=mono_f, relief="flat",
                        state="disabled", wrap="word", width=26, height=10)
    log_text.pack(fill="both", expand=True)

    # Footer
    tk.Frame(root, bg="#e5e5ea", height=1).pack(fill="x")
    footer = tk.Frame(root, bg="white", pady=8, padx=24)
    footer.pack(fill="x")
    tk.Label(footer,
             text='Use buttons to start/stop  |  Press F key to simulate fall detection',
             font=small_f, bg="white", fg="#8e8e93").pack(side="left")
    fps_label = tk.Label(footer, text="FPS: --", font=small_f,
                          bg="white", fg="#c7c7cc")
    fps_label.pack(side="right")

    root.bind('<f>', trigger_fall_alert)

    cap = cv2.VideoCapture(1)
    threading.Thread(target=voice_listener, daemon=True).start()
    add_log("BlindAssist v2.0 started.")
    add_log(f"Contact: {emergency_contact['name']} — {emergency_contact['phone']}")
    add_log("Press F key to simulate fall detection.")
    root.after(100, update)
    root.mainloop()
    cap.release()

# ─── Entry Point ─────────────────────────────────────────────────────────────
show_onboarding()
