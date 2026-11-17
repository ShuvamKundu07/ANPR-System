# ANPR Project POC --- Step-by-Step Implementation Guide for Antigravity

## 1. Project Overview

**Project:** Real-Time Automatic License Plate Recognition and
Intelligent Vehicle Tracking System (ANPR).

**Goal:** Build a working proof of concept that accepts a video/webcam
feed, detects license plates, reads registration text, stores
observations, checks a hotlist, and displays records and alerts.

Develop incrementally. Antigravity must complete one phase, run relevant
tests, report results, and wait for approval before moving on. Do not
generate the entire application in one step.

### POC success criteria

1.  Start the application and open a dashboard.
2.  Select a prerecorded vehicle video or webcam.
3.  Detect a license plate and draw its bounding box.
4.  Crop and save the detected plate image.
5.  Extract plate text using OCR and show OCR confidence.
6.  Store the observation in SQLite with timestamp, camera ID, image
    path, and recognized text.
7.  Compare normalized plate text against a sample hotlist.
8.  Display an alert when a hotlist match is found.
9.  Search and view historical vehicle observations.
10. Demonstrate basic two-camera movement history using the same plate
    number observed by two simulated camera sources.

**Scope note:** This demonstrates number-plate-based cross-camera
observation history. It is not production-grade multi-camera visual
tracking or vehicle re-identification.

## 2. Technology Stack

-   Python 3.10 or 3.11 (choose a version supported by the selected ML
    dependencies).
-   OpenCV --- video/image capture, frame processing, drawing boxes,
    cropping.
-   Ultralytics YOLO --- license-plate detection.
-   EasyOCR --- optical character recognition.
-   SQLite --- local persistent database.
-   Streamlit --- dashboard.
-   Pandas --- displaying/filtering records.
-   pytest --- automated tests.

### Important model requirement

A generic YOLO model trained on everyday objects may not recognize
license plates. Use a suitable license-plate detection model with a
known source and license. Record model name, source, version, expected
input size, and license in `models/README.md`.

If no suitable model is available, stop and ask the user to
provide/approve one. Do not fabricate weights or download from unknown
sources. A temporary placeholder detector or manually supplied plate
crops may be used to build other pipeline modules, but label this
limitation clearly.

## 3. Repository Structure

``` text
anpr-poc/
├── app.py
├── dashboard/
│   ├── __init__.py
│   └── views.py
├── anpr/
│   ├── __init__.py
│   ├── config.py
│   ├── video.py
│   ├── detector.py
│   ├── ocr.py
│   ├── plate_utils.py
│   ├── database.py
│   ├── hotlist.py
│   ├── alerts.py
│   └── tracking.py
├── models/
│   └── README.md
├── data/
│   ├── input_videos/
│   ├── plate_crops/
│   └── anpr.db
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

Keep model weights and large datasets out of Git unless explicitly
agreed. Add them to `.gitignore` and explain how to obtain/place them.

## 4. Rules for Antigravity

1.  Work only on the current phase. Do not skip ahead.
2.  Inspect the existing repository before creating or changing files.
3.  Preserve working code and avoid rewriting unrelated files.
4.  Use small, understandable modules and clear function names.
5.  Add type hints and concise docstrings to important functions.
6.  Keep model loading separate from frame processing.
7.  Do not hardcode absolute paths, credentials, or machine-specific
    settings.
8.  Configure model path, confidence threshold, camera ID, output
    directories, and duplicate-suppression interval.
9.  Handle missing model files, unavailable cameras, unreadable videos,
    empty OCR results, and database errors gracefully.
10. Never claim a feature works unless it has been run and tested.
11. Do not invent test results, model accuracy, or performance figures.
12. Use synthetic/sample hotlist data. Do not use real personal vehicle
    data without authorization.
13. Avoid creating a database record for every consecutive frame showing
    the same vehicle. Implement a configurable duplicate-suppression
    window.
14. At the end of each phase, report files changed, commands run, test
    results, known issues, and the exact next step.
15. Stop and ask the user before moving to the next phase.

## 5. Phase-by-Phase Implementation

## Phase 0 --- Inspect the Environment and Plan

**Objective:** Understand the development environment and create a safe
plan.

Tasks: 1. Inspect workspace, operating system, Python version, and
existing project files. 2. Do not delete or overwrite existing files. 3.
Check whether a virtual environment exists. 4. Check whether Git is
initialized. 5. Recommend Python 3.10 or 3.11 if the current version is
incompatible with ML packages. 6. Identify blockers, especially
availability of a trained license-plate detector. 7. Present findings
and ask for approval before substantial changes.

**Acceptance criteria:** Environment and files are summarized; nothing
is deleted; Python/environment choice is agreed.

**Stop and ask for approval.**

## Phase 1 --- Project Skeleton and Environment

**Objective:** Create folders, dependency list, configuration, and setup
instructions.

Tasks: 1. Create the repository structure above. 2. Create a Python
virtual environment. 3. Add compatible versions of `opencv-python`,
`ultralytics`, `easyocr`, `streamlit`, `pandas`, `pillow`, and `pytest`
to `requirements.txt`. 4. Add `.gitignore` for virtual environments,
caches, local database, generated crops, model weights, and secrets. 5.
Add configuration for model path, detector confidence threshold, default
camera ID, crop directory, and duplicate-suppression interval. 6. Create
a minimal README with installation and run commands. 7. Verify imports
in the selected environment.

Suggested commands:

``` bash
python -m venv .venv
```

Activate:

``` bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Acceptance criteria:** Project folders exist; dependencies install;
key imports work; README explains setup.

**Stop and ask for approval.**

## Phase 2 --- Video and Image Input with OpenCV

**Objective:** Read a local video or webcam and expose frames to later
modules.

Tasks: 1. Implement `anpr/video.py`. 2. Provide a function to open a
video file or camera index. 3. Implement a safe frame-reading loop. 4.
Handle invalid paths, unavailable cameras, and end-of-file. 5. Ensure
capture resources are released. 6. Add a small manual verification
script or test that displays/reads frames. 7. Keep this module
independent of YOLO and OCR.

**Acceptance criteria:** Sample video frames are read; invalid paths
produce clear errors; capture is released cleanly.

**Stop and ask for approval.**

## Phase 3 --- License Plate Detection with YOLO

**Objective:** Detect license plates and return bounding boxes with
confidence scores.

Tasks: 1. Implement `anpr/detector.py`. 2. Load approved license-plate
detection weights from configured path. 3. Verify model file exists
before loading. 4. Accept an image/frame and return zero or more
detections. 5. Each detection includes x1, y1, x2, y2, confidence, and
class label if available. 6. Draw boxes and confidence labels on a copy
of the frame. 7. Crop detected plates and save them under
`data/plate_crops/`. 8. Clamp crop coordinates to image boundaries and
reject invalid/empty crops. 9. Make confidence threshold configurable.
10. Record model source and license in `models/README.md`.

**Model handling:** Do not assume a general COCO YOLO model detects
plates. If no suitable model is available, stop and ask the user to
provide/approve one. Do not claim successful detection without testing.

**Acceptance criteria:** An approved sample with a visible plate
produces a plausible box; crop is saved and viewable; no-plate frame
returns an empty list safely; threshold is configurable.

**Stop and ask for approval.**

## Phase 4 --- OCR and Plate Text Normalization

**Objective:** Read the cropped plate and normalize its text for
matching.

Tasks: 1. Implement `anpr/ocr.py` with EasyOCR. 2. Initialize the OCR
reader once and reuse it. 3. Return raw OCR text and OCR confidence. 4.
Handle empty, unreadable, or low-confidence results. 5. Implement
`anpr/plate_utils.py`. 6. Normalize text to uppercase and remove spaces
and non-alphanumeric separators. 7. Preserve raw OCR output as well as
normalized output. 8. Add optional basic Indian registration format
validation as a heuristic, not proof of genuine registration. 9. Do not
silently replace ambiguous characters such as O/0 or I/1. If correction
heuristics are added, preserve the original and document the rule. 10.
Add unit tests for normalization and validation.

**Acceptance criteria:** Clear sample crop returns readable text;
confidence is returned; empty OCR is safe; normalization tests pass; raw
and normalized values remain distinguishable.

**Stop and ask for approval.**

## Phase 5 --- SQLite Database

**Objective:** Store observations and retrieve them for the dashboard.

Create these tables:

**`vehicle_records`** - `id` --- integer primary key - `plate_text_raw`
--- text - `plate_number` --- normalized text - `ocr_confidence` ---
real - `detected_at` --- timestamp in a consistent format - `camera_id`
--- text - `image_path` --- text - `detection_confidence` --- real

**`hotlist`** - `id` --- integer primary key - `plate_number` --- unique
normalized text - `status` --- text - `description` --- optional text -
`created_at` --- timestamp

**`alerts`** - `id` --- integer primary key - `vehicle_record_id` ---
foreign key to vehicle_records - `plate_number` --- text - `alert_type`
--- text - `message` --- text - `created_at` --- timestamp -
`acknowledged` --- boolean/integer

Tasks: 1. Implement `anpr/database.py`. 2. Create tables automatically
on first start. 3. Use parameterized SQL queries. 4. Add functions to
insert observations, fetch recent records, search by plate, and retrieve
movement history. 5. Add functions to add/list/remove hotlist entries.
6. Add functions to create/list alerts. 7. Store image paths relative to
project where practical. 8. Test with a temporary test database.

**Acceptance criteria:** Database initializes; sample observation can be
inserted/retrieved; search works; tests do not modify the real database.

**Stop and ask for approval.**

## Phase 6 --- Hotlist Matching and Alerts

**Objective:** Match recognized plate numbers against the hotlist and
generate alerts.

Tasks: 1. Implement `anpr/hotlist.py`. 2. Normalize incoming plate using
the same function used for hotlist entries. 3. Compare exact normalized
strings first. 4. Do not use fuzzy matching by default; incorrect fuzzy
matches can create false alerts. 5. Implement `anpr/alerts.py`. 6. On
exact match, create an alert linked to the vehicle observation. 7.
Include plate, hotlist status, camera ID, timestamp, and snapshot path.
8. Prevent repeated alerts for the same observation across consecutive
frames. 9. Test match, non-match, duplicate observations, and alert
creation.

**Acceptance criteria:** Hotlisted sample generates alert; non-hotlisted
plate does not; duplicate alerts are controlled; alerts are stored in
SQLite.

**Stop and ask for approval.**

## Phase 7 --- End-to-End ANPR Pipeline

**Objective:** Connect video, detection, OCR, database, and hotlist
matching.

Tasks: 1. Implement orchestration in `app.py` or a dedicated pipeline
module. 2. Read a frame from selected video/camera. 3. Run plate
detection and crop each plate. 4. Run OCR on each crop. 5. Normalize and
validate recognized text. 6. Apply configurable OCR confidence
threshold. 7. Store valid observations in SQLite. 8. Check hotlist and
create alerts where appropriate. 9. Draw plate box and recognized text
on displayed frame. 10. Implement duplicate suppression using camera ID,
normalized plate, and configurable time window. 11. Avoid reinitializing
models for every frame. 12. Keep processing responsive and log errors
clearly.

**Acceptance criteria:** Sample video runs through complete pipeline;
annotated frame and OCR result are visible; records are stored; hotlist
match creates alert; repeated frames do not flood database.

**Stop and ask for approval.**

## Phase 8 --- Streamlit Dashboard

**Objective:** Build a dashboard for live monitoring, records, hotlist,
and alerts.

### Live Monitoring

-   Choose a video file or webcam.
-   Set camera ID.
-   Start and stop processing.
-   Show annotated frames.
-   Show latest recognized plate and confidence.

### Vehicle Records

-   Show recent observations.
-   Search by plate.
-   Filter by camera and date where practical.
-   Open saved plate crop.

### Hotlist Management

-   Add plate number and status.
-   List current entries.
-   Remove an entry.
-   Normalize values before saving.

### Alerts

-   Show hotlist matches.
-   Display status, timestamp, camera ID, and plate image.
-   Allow an alert to be acknowledged.

### Movement History

-   Search a plate.
-   Display observations ordered by timestamp.
-   Show camera IDs/locations as a sequence.

UI requirements: - Keep interface clean and suitable for academic
demonstration. - Show loading and error states. - Keep Streamlit UI
separate from detection, OCR, and database logic.

**Acceptance criteria:** Dashboard starts with documented command; each
section reads/writes SQLite correctly; live monitoring starts/stops;
hotlist and alert actions work.

**Stop and ask for approval.**

## Phase 9 --- Two-Camera Movement History Demonstration

**Objective:** Demonstrate basic plate-based movement history across two
camera sources.

This is a controlled simulation, not full visual multi-camera tracking.

Tasks: 1. Configure two camera IDs, e.g. `CAM001` and `CAM002`. 2. Use
two prerecorded clips or input sources. 3. Process each source and store
observations with its camera ID. 4. Use normalized plate number as
matching key. 5. Search a plate and show observations ordered by
timestamp. 6. Display a sample sequence: - CAM001 --- Location A ---
10:30 - CAM002 --- Location B --- 10:42 7. Label this as inferred
observation history based on plate recognition. 8. Do not claim a
continuous trajectory or prove physical identity if OCR may be wrong.

**Acceptance criteria:** Same sample plate from both sources appears in
one ordered history; camera IDs/timestamps are visible; different plates
are not merged; limitation is explained in UI.

**Stop and ask for approval.**

## Phase 10 --- Testing, Evaluation, and Reliability

**Objective:** Verify the POC and report measured performance honestly.

Test: 1. Clear plate in daylight. 2. Plate at an angle. 3.
Blurry/low-resolution plate. 4. Low-light/night sample. 5. Frame with no
visible plate. 6. Plate present in hotlist. 7. Plate absent from
hotlist. 8. Same plate repeated across many frames. 9. Invalid video
path or unavailable webcam. 10. Database unavailable or output directory
unwritable.

Measure and report: - Number of test images/videos. - Detection
precision and recall, if ground-truth labels are available. - Exact
plate recognition accuracy. - Character-level OCR accuracy, if
measured. - Hotlist matching correctness. - Average processing time per
frame. - False detections and false alerts.

Do not state target accuracy as achieved unless the test data and
calculations support it. Separate daylight/night results only if enough
samples exist.

**Acceptance criteria:** Automated tests pass; manual scenarios are
documented; errors are handled; measured results are added to README.

**Stop and ask for approval.**

## Phase 11 --- Final Demo and Documentation

**Objective:** Prepare a repeatable academic demonstration.

Demo sequence: 1. Start Streamlit. 2. Open Live Monitoring. 3. Load a
sample vehicle video. 4. Show a detected plate bounding box. 5. Show
cropped plate and OCR output. 6. Show observation in Vehicle Records. 7.
Add a sample plate to hotlist. 8. Process a clip containing that plate.
9. Show generated alert. 10. Search the plate and show history across
two simulated camera IDs.

README must include: - Project overview and POC scope. - Architecture
and data flow. - Technology stack and prerequisites. - Installation for
Windows and macOS/Linux. - Model source, version, and license. - How to
run the app and tests. - Sample data instructions. - Known
limitations. - Measured evaluation results. - Future enhancements.

**Acceptance criteria:** A new team member can install and run the POC
from README; demo is repeatable without manual database edits;
limitations are documented.

## 6. Suggested Team Division

For a four-person team:

  Team member   Responsibility
  ------------- --------------------------------------------------
  Member 1      YOLO model selection, setup, and plate detection
  Member 2      OCR, text normalization, and validation
  Member 3      SQLite, hotlist matching, alerts, and tests
  Member 4      Streamlit dashboard and integration

All members should understand the complete pipeline.

## 7. Final Definition of Done

-   [ ] Video/webcam input works.
-   [ ] Suitable license-plate model detects plates.
-   [ ] Plate crops are generated and saved.
-   [ ] OCR returns text and confidence.
-   [ ] Text normalization and validation work.
-   [ ] Vehicle observations are stored in SQLite.
-   [ ] Hotlist matching works.
-   [ ] Alerts are generated and displayed.
-   [ ] Dashboard pages work.
-   [ ] Two-camera plate-based observation history is demonstrated.
-   [ ] Tests and measured results are documented.
-   [ ] Setup and demo instructions are reproducible.

## 8. First Prompt to Give Antigravity

Copy and paste this prompt into Antigravity:

> We are building a working Proof of Concept for a Real-Time Automatic
> License Plate Recognition and Intelligent Vehicle Tracking System
> (ANPR). Follow the implementation guide in this Markdown file exactly
> and work phase by phase.
>
> Start with **Phase 0 only**. Inspect the current workspace, operating
> system, Python version, existing files, and environment. Do not delete
> or overwrite anything. Identify blockers, especially whether a
> suitable trained license-plate YOLO model is available. Report
> findings and propose the Phase 1 setup.
>
> Do not implement later phases yet. Do not claim anything has been
> tested unless you actually ran the test. After Phase 0, stop and wait
> for my approval.

## 9. Prompt Template for Each Next Phase

After reviewing and approving a phase, use:

> Continue with the next incomplete phase in
> `ANPR_POC_IMPLEMENTATION_GUIDE.md`. Read that phase carefully and
> implement only that phase. Inspect existing code before editing,
> preserve working functionality, write or update relevant tests, run
> the tests, and report changed files, commands executed, actual
> results, known issues, and acceptance-criteria status. Do not move to
> the following phase until I approve.

------------------------------------------------------------------------

**Development principle:** Build a small working pipeline first.
Prioritize reliable plate detection → OCR → database → hotlist alert.
Add dashboard polish and simulated multi-camera history after the core
flow works.
