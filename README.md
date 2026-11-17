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

## Architecture & Data Flow

```text
[ Video / Webcam Feed ] (anpr.video)
          │
          ▼
[ YOLO Plate Detector ] (anpr.detector)  ───> Crop Plate Image (data/plate_crops/)
          │
          ▼
[ EasyOCR Text Extraction ] (anpr.ocr)
          │
          ▼
[ Plate Normalizer & Validator ] (anpr.plate_utils)
          │
          ▼
[ Duplicate Suppression Engine ] (anpr.tracking)
     │                     │
 (New Observation)     (Within Cooldown)
     ▼                     ▼
[ SQLite Database ]    [ Skipped ]
  • vehicle_records
  • hotlist
  • alerts
     │
     ▼
[ Hotlist Matcher & Alert Dispatch ] (anpr.hotlist, anpr.alerts)
     │
     ▼
[ Streamlit Live Dashboard ] (dashboard.views / app.py)
```

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

# Run two-camera simulated journey demo (Phase 9)
python scripts/simulate_two_cameras.py --max-frames 150

# Run Phase 10 quantitative benchmark evaluation
python scripts/evaluate_reliability.py
```

Open <http://localhost:8501> in your browser.

---

## Running Tests

```bash
# Run all unit, integration, and reliability test suites
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

## Measured Evaluation & Benchmark Results (Phase 10)

Evaluation performed using `scripts/evaluate_reliability.py` across diverse lighting, angle, and distortion conditions (Apple Silicon CPU):

| Metric | Measured Value | Notes |
|---|---|---|
| **Test Cases Evaluated** | **13** | 8 vehicle conditions + 5 negative background frames |
| **Plate Detection Recall** | **100.0%** (8/8) | YOLO detector localized all plates across all conditions |
| **Plate Detection Precision** | **100.0%** (8/8) | Zero false-positive detections on blank backgrounds |
| **False Detections** | **0 / 5** | No phantom plate boxes on non-vehicle frames |
| **Exact Plate OCR Accuracy** | **25.0%** (2/8) | Exact string match (clear daylight samples) |
| **Character-Level OCR Accuracy**| **73.4%** (58/79) | Matches across angles, blur, and lighting conditions |
| **Hotlist Trigger Accuracy** | **33.3%** (1/3) | Strict exact normalized matching prevents false alerts |
| **False Hotlist Alerts** | **0 (0.0%)** | Zero unflagged vehicles triggered hotlist alerts |
| **Average Processing Latency** | **67.4 ms/frame** | Measured end-to-end on CPU (~14.8 FPS) |
| **Duplicate Suppression** | **100% effective** | Suppresses repeated consecutive observations cleanly |

---

## Known Limitations

- **OCR Ambiguities:** Character confusion occurs under blur or rotation (e.g. `O` vs `0`, `I` vs `1`, `Z` vs `2`). Heuristic post-processing is intentionally transparent rather than silently guessing.
- **Cross-Camera Association:** Journey tracking links observations by normalized OCR text only. It does not provide continuous physical trajectory or vehicle re-identification.
- **Processing Speed:** CPU inference averages ~14.8 FPS; GPU acceleration or model quantization is recommended for production 30+ FPS feeds.

---

## Sample Data & Demo Walkthrough (Phase 11)

### Sample Data Placement
- Place demo video files in `data/input_videos/` (e.g. `sample_test.mp4`).
- Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`.

### Repeatable Academic Demo Sequence
1. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```
2. **Live Monitoring:**
   - Navigate to `📹 Live Monitoring` in the sidebar.
   - Choose a video file or webcam, set Camera ID to `CAM001`, and click **Start**.
   - Observe real-time bounding boxes drawn on detected plates and live OCR confidence tags.
3. **Vehicle Records:**
   - Navigate to `📋 Vehicle Records`.
   - Inspect stored observations with timestamps, raw OCR, normalized plate numbers, and view the saved high-resolution plate crops.
4. **Hotlist & Alerts:**
   - Navigate to `🚨 Hotlist`.
   - Add a plate number (e.g. `DL8CAB1234` or a plate appearing in your test video) with status `STOLEN`.
   - In Live Monitoring, re-process the clip or feed.
   - Navigate to `🔔 Alerts` to verify the red alert banner, snapshot, and click **Acknowledge**.
5. **Cross-Camera Journey History:**
   - Navigate to `🎬 Two-Camera Demo`.
   - Run the simulation to process a clip across `CAM001` (Sector 12) and `CAM002` (Sector 14) with staggered timestamps.
   - View the chronological observation journey and cross-camera summary table.
   - Alternatively, search any plate under `🗺️ Movement History`.

---

## Future Enhancements

- GPU-accelerated inference
- Real-time WebRTC camera feed
- Vehicle make / colour recognition
- Production multi-camera visual re-identification
- REST API for integration with external systems
