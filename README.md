# Fire Detection & Alarm System 🔥🚨

An automated real-time fire detection and alarm application built with Python, OpenCV, and deep learning models (**YOLO / ONNX**), designed to detect fires instantly and trigger audio alerts.

---

## 📌 Features
- **Real-time Fire Detection:** Detects fire hazards in images or video streams using advanced deep learning models (`.pt` and `.onnx`).
- **Automated Alarm System:** Plays an instant audio warning (`fire_alarm.mp3`) when a fire is detected.
- **Model Export Support:** Includes utilities to export and run models efficiently.

---

## 🛠️ Project Structure
```text
Fire_Detection/
├── Images/ (or project files)
├── Alarm_Player.py          # Script for handling sound/audio alerts
├── best.pt                  # YOLO trained model weights
├── best.onnx                # Exported ONNX model format
├── export_onnx.py           # Script for converting models to ONNX
├── fire_alarm.mp3           # Alarm sound effect
├── fire_detection_cascade_model.xml  # Haar cascade model (if applicable)
├── onnx_program_run.py      # Main script to run detection using ONNX
├── program_run.py           # Main execution script
└── README.md                # Project documentation

## 🚀 Getting Started
1. Prerequisites
Ensure you have Python installed, along with the required libraries:
```bash
pip install opencv-python numpy

2. Running the Detection Program
To run the fire detection and alarm system:
'''python program_run.py

(Ory use the ONNX runner script):
'''python onnx_program_run.py

## 👤 Author
- **Hasan Srour** - [GitHub Profile](https://github.com/EngHasanSrour)
