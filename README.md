## 🚀 New Features (ML-Enhanced)

### Depth Estimation System
- **Model:** MiDaS (Monocular Depth Estimation)
- **Function:** Estimates distance to detected objects
- **Output:** "Chair 2.5 meters ahead"

### Smart Audio Navigation
- **Intelligent Priority System:** Announces critical obstacles first
- **Distance-Based Urgency:** "Warning! Person very close" vs "Chair 5 meters ahead"
- **Anti-Spam:** Cooldown system prevents repetitive announcements

### Tech Stack
- YOLOv8: Object detection (80 classes)
- MiDaS DPT-Large: Depth estimation
- PyTorch: Deep learning framework
- pyttsx3: Text-to-speech engine

## 📦 Installation

```bash
git clone https://github.com/preranaprakash129-netizen/blindassist.git
cd blindassist
pip install -r requirements.txt
```

## ▶️ Usage

```bash
cd src
python main_navigation_app.py
```

Press 'q' to quit.