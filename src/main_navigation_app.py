import sys
import cv2
import time
import numpy as np
import threading
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QPushButton, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QPainter, QLinearGradient

from depth_yolo_pipeline import DepthYOLOPipeline
from audio_feedback import AudioFeedbackSystem

# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
BG_DARK      = "#0a0e1a"
BG_PANEL     = "#111827"
BG_CARD      = "#1a2235"
ACCENT_BLUE  = "#3b82f6"
ACCENT_CYAN  = "#06b6d4"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED   = "#ef4444"
TEXT_PRIMARY = "#f1f5f9"
TEXT_MUTED   = "#64748b"
BORDER       = "#1e293b"

CAUTION_COLORS = {
    "low":    ACCENT_GREEN,
    "medium": ACCENT_AMBER,
    "high":   ACCENT_RED,
}

SCENE_ICONS = {
    "indoor":    "🏠",
    "outdoor":   "🌳",
    "staircase": "🪜",
    "corridor":  "🚪",
    "office":    "💼",
    "crowded":   "👥",
}


# ─────────────────────────────────────────────
#  WORKER THREAD  (keeps UI responsive)
# ─────────────────────────────────────────────
class ProcessingWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, list, str, dict, float)
    error       = pyqtSignal(str)

    def __init__(self, pipeline, audio_system, camera_index=0):
        super().__init__()
        self.pipeline     = pipeline
        self.audio_system = audio_system
        self.camera_index = camera_index
        self.running      = False
        self.process_every = 5
        self._frame_count  = 0

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit("Could not open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.error.emit("Failed to grab frame.")
                break

            self._frame_count += 1
            if self._frame_count % self.process_every == 0:
                t0 = time.time()
                detections, depth_map, scene_type, scene_context = \
                    self.pipeline.process_frame(frame)
                fps = 1.0 / max(time.time() - t0, 1e-6)

                self.audio_system.process_detections(detections, scene_context)
                self.frame_ready.emit(frame, depth_map, detections,
                                      scene_type, scene_context, fps)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ─────────────────────────────────────────────
#  TINY REUSABLE WIDGETS
# ─────────────────────────────────────────────
def _label(text, font_size=11, bold=False, color=TEXT_PRIMARY, align=Qt.AlignLeft):
    lbl = QLabel(text)
    lbl.setAlignment(align)
    w = "600" if bold else "400"
    lbl.setStyleSheet(f"color:{color}; font-size:{font_size}px; font-weight:{w};")
    return lbl


def _hline():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color:{BORDER};")
    return line


class BadgeLabel(QLabel):
    """Pill-shaped coloured badge."""
    def __init__(self, text="", color=ACCENT_BLUE, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.set_color(color)

    def set_color(self, color):
        self.setStyleSheet(
            f"background:{color}22; color:{color}; border:1px solid {color}55;"
            f"border-radius:10px; padding:2px 10px; font-size:11px; font-weight:600;"
        )


class DetectionCard(QFrame):
    """One row in the detections panel."""
    def __init__(self, det, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background:{BG_CARD}; border:1px solid {BORDER};"
            f"border-radius:8px; margin:2px 0;"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)

        name = det["class"].replace("_", " ").title()
        dist = det["distance_m"]
        conf = det["confidence"]

        # Urgency colour
        if dist is not None:
            if dist < 1.5:
                col = ACCENT_RED
            elif dist < 3.0:
                col = ACCENT_AMBER
            else:
                col = ACCENT_GREEN
        else:
            col = TEXT_MUTED

        dot = QLabel("●")
        dot.setStyleSheet(f"color:{col}; font-size:10px;")
        lay.addWidget(dot)

        name_lbl = _label(name, font_size=12, bold=True)
        lay.addWidget(name_lbl, stretch=1)

        if dist is not None:
            dist_lbl = BadgeLabel(f"{dist:.1f} m", col)
        else:
            dist_lbl = BadgeLabel("—", TEXT_MUTED)
        lay.addWidget(dist_lbl)

        conf_lbl = _label(f"{conf*100:.0f}%", color=TEXT_MUTED, font_size=10,
                          align=Qt.AlignRight)
        lay.addWidget(conf_lbl)


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────
class BlindAssistUI(QMainWindow):
    def __init__(self, pipeline, audio_system):
        super().__init__()
        self.pipeline     = pipeline
        self.audio_system = audio_system
        self._fps_history = deque(maxlen=30)

        self.setWindowTitle("BlindAssist — Navigation Dashboard")
        self.setMinimumSize(1280, 760)
        self._apply_global_style()
        self._build_ui()
        self._start_worker()

    # ── styling ──────────────────────────────
    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{BG_DARK}; color:{TEXT_PRIMARY}; font-family:'Segoe UI', sans-serif; }}
            QScrollArea {{ border:none; background:transparent; }}
            QScrollBar:vertical {{ background:{BG_PANEL}; width:6px; border-radius:3px; }}
            QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:3px; }}
            QPushButton {{
                background:{BG_CARD}; color:{TEXT_PRIMARY}; border:1px solid {BORDER};
                border-radius:8px; padding:8px 18px; font-size:12px;
            }}
            QPushButton:hover {{ background:{ACCENT_BLUE}22; border-color:{ACCENT_BLUE}; }}
            QPushButton:pressed {{ background:{ACCENT_BLUE}44; }}
        """)

    # ── layout skeleton ───────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(16, 16, 16, 16)
        root_lay.setSpacing(12)

        root_lay.addWidget(self._build_header())
        root_lay.addWidget(self._build_status_bar())

        body_lay = QHBoxLayout()
        body_lay.setSpacing(12)
        body_lay.addLayout(self._build_feeds(),    stretch=3)
        body_lay.addWidget(self._build_sidebar(),  stretch=1)
        root_lay.addLayout(body_lay, stretch=1)

        root_lay.addWidget(self._build_footer())

    # ── header ───────────────────────────────
    def _build_header(self):
        frame = QFrame()
        frame.setStyleSheet(
            f"background:{BG_PANEL}; border:1px solid {BORDER}; border-radius:12px;"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(20, 12, 20, 12)

        title = QLabel("👁  BlindAssist")
        title.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:20px; font-weight:700; letter-spacing:1px;"
        )
        lay.addWidget(title)
        lay.addStretch()

        self.fps_badge  = BadgeLabel("FPS: —", ACCENT_CYAN)
        self.audio_badge = BadgeLabel("🔇 Audio idle", TEXT_MUTED)
        lay.addWidget(self.fps_badge)
        lay.addSpacing(8)
        lay.addWidget(self.audio_badge)
        lay.addSpacing(16)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.clicked.connect(self._stop)
        lay.addWidget(self.stop_btn)

        return frame

    # ── status / scene bar ────────────────────
    def _build_status_bar(self):
        frame = QFrame()
        frame.setStyleSheet(
            f"background:{BG_PANEL}; border:1px solid {BORDER}; border-radius:10px;"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(24)

        # Scene
        self.scene_icon  = _label("🏠", font_size=18)
        self.scene_label = _label("Scene: —", font_size=12, bold=True)
        self.caution_badge = BadgeLabel("low", ACCENT_GREEN)

        lay.addWidget(self.scene_icon)
        lay.addWidget(self.scene_label)
        lay.addWidget(self.caution_badge)
        lay.addSpacing(8)
        lay.addWidget(_hline())
        lay.addSpacing(8)

        # Warning message
        self.warning_label = _label("No warnings", color=TEXT_MUTED, font_size=11)
        lay.addWidget(self.warning_label, stretch=1)

        # Detection count
        self.det_count_label = _label("Objects: 0", color=ACCENT_CYAN,
                                      font_size=12, bold=True, align=Qt.AlignRight)
        lay.addWidget(self.det_count_label)
        return frame

    # ── dual camera feeds ─────────────────────
    def _build_feeds(self):
        lay = QVBoxLayout()
        lay.setSpacing(8)

        feeds_lay = QHBoxLayout()
        feeds_lay.setSpacing(8)

        # Camera feed
        cam_card = self._feed_card("📷  Live Camera")
        self.cam_label = cam_card[1]
        feeds_lay.addWidget(cam_card[0])

        # Depth map
        depth_card = self._feed_card("🌊  Depth Map")
        self.depth_label = depth_card[1]
        feeds_lay.addWidget(depth_card[0])

        lay.addLayout(feeds_lay)
        return lay

    def _feed_card(self, title):
        frame = QFrame()
        frame.setStyleSheet(
            f"background:{BG_PANEL}; border:1px solid {BORDER}; border-radius:12px;"
        )
        vlay = QVBoxLayout(frame)
        vlay.setContentsMargins(10, 10, 10, 10)
        vlay.setSpacing(6)

        title_lbl = _label(title, font_size=11, bold=True, color=TEXT_MUTED)
        vlay.addWidget(title_lbl)

        feed = QLabel()
        feed.setAlignment(Qt.AlignCenter)
        feed.setMinimumSize(420, 320)
        feed.setStyleSheet(
            f"background:#000; border-radius:8px; color:{TEXT_MUTED}; font-size:13px;"
        )
        feed.setText("Waiting for camera…")
        vlay.addWidget(feed, stretch=1)
        return frame, feed

    # ── right sidebar ─────────────────────────
    def _build_sidebar(self):
        frame = QFrame()
        frame.setStyleSheet(
            f"background:{BG_PANEL}; border:1px solid {BORDER}; border-radius:12px;"
        )
        vlay = QVBoxLayout(frame)
        vlay.setContentsMargins(12, 12, 12, 12)
        vlay.setSpacing(10)

        vlay.addWidget(_label("📋  Detected Objects", font_size=13, bold=True))
        vlay.addWidget(_hline())

        # Scrollable detections list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.det_container = QWidget()
        self.det_layout    = QVBoxLayout(self.det_container)
        self.det_layout    .setContentsMargins(0, 0, 0, 0)
        self.det_layout    .setSpacing(4)
        self.det_layout    .addStretch()

        scroll.setWidget(self.det_container)
        vlay.addWidget(scroll, stretch=1)

        vlay.addWidget(_hline())
        vlay.addWidget(_label("⚡  Scene Confidence", font_size=11, color=TEXT_MUTED))
        self.scene_conf_label = _label("—", font_size=22, bold=True,
                                       color=ACCENT_BLUE, align=Qt.AlignCenter)
        vlay.addWidget(self.scene_conf_label)
        return frame

    # ── footer ───────────────────────────────
    def _build_footer(self):
        frame = QFrame()
        frame.setStyleSheet(
            f"background:{BG_PANEL}; border:1px solid {BORDER}; border-radius:10px;"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 6, 16, 6)
        self.footer_label = _label(
            "YOLOv8 • MiDaS DPT-Large • MobileNetV2  |  BlindAssist v1.0",
            color=TEXT_MUTED, font_size=10
        )
        lay.addWidget(self.footer_label)
        lay.addStretch()
        self.frame_count_label = _label("Frame: 0", color=TEXT_MUTED, font_size=10,
                                        align=Qt.AlignRight)
        lay.addWidget(self.frame_count_label)
        return frame

    # ── worker ────────────────────────────────
    def _start_worker(self):
        self.worker = ProcessingWorker(self.pipeline, self.audio_system,camera_index=1)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        self._frame_idx = 0

    # ── frame update ─────────────────────────
    def _on_frame(self, frame, depth_map, detections, scene_type, scene_context, fps):
        self._frame_idx += 1
        self._fps_history.append(fps)
        avg_fps = sum(self._fps_history) / len(self._fps_history)

        # ── camera feed ──
        annotated = self._draw_boxes(frame.copy(), detections)
        self.cam_label.setPixmap(self._to_pixmap(annotated, self.cam_label.size()))

        # ── depth map ──
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        self.depth_label.setPixmap(self._to_pixmap(depth_color, self.depth_label.size()))

        # ── header badges ──
        self.fps_badge.setText(f"FPS: {avg_fps:.1f}")
        if self.audio_system.speaking:
            self.audio_badge.setText("🔊 Speaking")
            self.audio_badge.set_color(ACCENT_CYAN)
        else:
            self.audio_badge.setText("🔇 Audio idle")
            self.audio_badge.set_color(TEXT_MUTED)

        # ── status bar ──
        icon = SCENE_ICONS.get(scene_type, "📍")
        self.scene_icon.setText(icon)
        self.scene_label.setText(f"Scene: {scene_type.title()}")

        caution = scene_context.get("caution_level", "low")
        caution_col = CAUTION_COLORS.get(caution, ACCENT_GREEN)
        self.caution_badge.setText(caution.upper())
        self.caution_badge.set_color(caution_col)

        warning = scene_context.get("warning", "")
        self.warning_label.setText(warning or "No warnings")
        self.warning_label.setStyleSheet(
            f"color:{caution_col if caution != 'low' else TEXT_MUTED}; font-size:11px;"
        )

        self.det_count_label.setText(f"Objects: {len(detections)}")

        # ── sidebar detections ──
        self._refresh_detections(detections)
        self.scene_conf_label.setText(f"{len(detections)} obj")

        # ── footer ──
        self.frame_count_label.setText(f"Frame: {self._frame_idx}")

    def _draw_boxes(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            dist = det["distance_m"]
            conf = det["confidence"]
            name = det["class"]

            if dist is not None:
                color = (60, 80, 239) if dist >= 3 else \
                        (15, 158, 245) if dist >= 1.5 else (68, 68, 239)
                # BGR
                color = (68, 68, 239) if dist < 1.5 else \
                        (15, 200, 245) if dist < 3.0 else (16, 185, 129)
            else:
                color = (100, 100, 100)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            lbl = f"{name} {dist:.1f}m" if dist else name
            lbl += f" {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, lbl, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return frame

    def _refresh_detections(self, detections):
        # Clear old cards (keep stretch at end)
        while self.det_layout.count() > 1:
            item = self.det_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Sort closest first
        sorted_dets = sorted(
            detections,
            key=lambda d: d["distance_m"] if d["distance_m"] is not None else 999
        )
        for det in sorted_dets[:12]:           # cap at 12 cards
            card = DetectionCard(det)
            self.det_layout.insertWidget(self.det_layout.count() - 1, card)

        if not detections:
            placeholder = _label("No objects detected", color=TEXT_MUTED,
                                  font_size=11, align=Qt.AlignCenter)
            self.det_layout.insertWidget(0, placeholder)

    @staticmethod
    def _to_pixmap(frame_bgr, target_size):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        return pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # ── error / stop ─────────────────────────
    def _on_error(self, msg):
        self.warning_label.setText(f"⚠ {msg}")
        self.warning_label.setStyleSheet(f"color:{ACCENT_RED}; font-size:11px;")

    def _stop(self):
        self.worker.stop()
        self.stop_btn.setText("⏹  Stopped")
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


# ─────────────────────────────────────────────
#  LOADING SCREEN
# ─────────────────────────────────────────────
class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BlindAssist — Loading")
        self.setFixedSize(480, 260)
        self.setStyleSheet(f"background:{BG_DARK};")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(16)

        title = QLabel("👁  BlindAssist")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:26px; font-weight:700; letter-spacing:2px;"
        )
        lay.addWidget(title)

        sub = QLabel("Navigation System for the Visually Impaired")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color:{TEXT_MUTED}; font-size:12px;")
        lay.addWidget(sub)

        lay.addSpacing(8)
        self.status = QLabel("Initialising…")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet(f"color:{ACCENT_CYAN}; font-size:13px;")
        lay.addWidget(self.status)

        credit = QLabel("YOLOv8 · MiDaS DPT-Large · MobileNetV2")
        credit.setAlignment(Qt.AlignCenter)
        credit.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        lay.addWidget(credit)

    def set_status(self, text):
        self.status.setText(text)
        QApplication.processEvents()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    # Show loading screen while models initialise
    loader = LoadingScreen()
    loader.show()

    loader.set_status("[1/3] Loading AI models…")
    pipeline = DepthYOLOPipeline(yolo_model_path="../models/yolov8.pt")

    loader.set_status("[2/3] Initialising audio system…")
    audio_system = AudioFeedbackSystem()

    loader.set_status("[3/3] Starting UI…")
    window = BlindAssistUI(pipeline, audio_system)
    loader.close()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()