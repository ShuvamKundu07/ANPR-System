# ANPR — Real-Time Automatic License Plate Recognition (POC)

A working proof-of-concept for automatic license-plate recognition and
plate-based vehicle observation history, built as an academic demonstration.

---

## Project Overview

Accepts a video file or webcam feed, detects license plates with a YOLO model,
reads plate text with EasyOCR, stores observations in SQLite, checks a hotlist,
and displays everything in a Streamlit dashboard.

**Scope:** Plate-number-based cross-camera observation history.  
Not production-grade multi-camera visual tracking or vehicle re-identification.

---

## Technology Stack

| Component      | Library               |
|----------------|-----------------------|
| Video / Image  | OpenCV 4.10           |
| Detection      | Ultralytics YOLO 8.3  |
| OCR            | EasyOCR 1.7           |
| Database       | SQLite (built-in)     |
| Dashboard      | Streamlit 1.41        |
| Data tables    | Pandas 2.2            |
| Tests          | pytest 8.3            |

---

## Prerequisites

- Python 3.11 or 3.13 (Apple Silicon: Homebrew `python@3.13` works well)
- A trained license-plate detection YOLO `.pt` weight file  
  (see `models/README.md` for the approved model, source, and license)

---

## Installation

```bash
# 1. Clone / download the repository
cd anpr-poc

# 2. Create a virtual environment
python3.13 -m venv .venv          # or python3.11

# 3. Activate
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Place your YOLO weights
#    Copy the approved .pt file to: models/license_plate_detector.pt
#    See models/README.md for details.

# 6. (Optional) Copy .env.example to .env and adjust settings
cp .env.example .env
```

---

## Running the Application

```bash
# Activate venv first
source .venv/bin/activate

# Launch the Streamlit dashboard
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
anpr-poc/
├── app.py                  # Streamlit entry point
├── dashboard/
│   ├── __init__.py
│   └── views.py            # Dashboard page renderers
├── anpr/
│   ├── __init__.py
│   ├── config.py           # Centralised settings
│   ├── video.py            # Video / webcam capture
│   ├── detector.py         # YOLO plate detector
│   ├── ocr.py              # EasyOCR wrapper
│   ├── plate_utils.py      # Text normalisation & validation
│   ├── database.py         # SQLite persistence
│   ├── hotlist.py          # Hotlist matching
│   ├── alerts.py           # Alert creation
│   └── tracking.py         # Duplicate suppression / observation tracker
├── models/
│   └── README.md           # Model source, version, and license
├── data/
│   ├── input_videos/       # Test / demo video clips (not in git)
│   ├── plate_crops/        # Saved plate crops (not in git)
│   └── anpr.db             # SQLite database (not in git)
├── tests/
│   ├── test_plate_utils.py
│   ├── test_database.py
│   ├── test_hotlist.py
│   └── test_pipeline.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Model Information

See `models/README.md` for the approved model name, download source, expected
input size, and license.

---

## Configuration

Copy `.env.example` to `.env` and edit values:

| Variable                  | Default                              | Description                        |
|---------------------------|--------------------------------------|------------------------------------|
| `MODEL_PATH`              | `models/license_plate_detector.pt`   | Path to YOLO weights               |
| `DETECTOR_CONFIDENCE`     | `0.4`                                | Minimum detection confidence       |
| `DEFAULT_CAMERA_ID`       | `0`                                  | Webcam index                       |
| `CROP_DIR`                | `data/plate_crops`                   | Where plate crops are saved        |
| `DB_PATH`                 | `data/anpr.db`                       | SQLite database path               |
| `DUPLICATE_WINDOW_SEC`    | `30`                                 | Seconds to suppress duplicate obs  |
| `OCR_CONFIDENCE_THRESHOLD`| `0.3`                                | Minimum OCR confidence             |

---

## Known Limitations

- OCR accuracy depends on plate clarity, angle, and lighting.
- Cross-camera "tracking" is based solely on matched plate text — not visual re-identification.
- No GPU required, but detection and OCR are slower on CPU-only machines.
- Detailed measured evaluation results will be added after Phase 10.

---

## Future Enhancements

- GPU-accelerated inference
- Real-time WebRTC camera feed
- Vehicle make / colour recognition
- Production multi-camera visual re-identification
- REST API for integration with external systems
