"""
anpr/config.py
──────────────
Centralised configuration for the ANPR POC.

All tuneable values are read from environment variables (via .env).
No absolute paths, credentials, or machine-specific defaults are hard-coded.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if it exists (safe to call even when .env is absent)
load_dotenv()

# ── Directories ──────────────────────────────────────────────────────────────
# Project root is two levels up from this file: anpr/config.py → anpr/ → root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

def _path(env_var: str, default: str) -> Path:
    """Return an absolute Path, resolved relative to PROJECT_ROOT."""
    raw = os.getenv(env_var, default)
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p

MODEL_PATH: Path = _path("MODEL_PATH", "models/license_plate_detector.pt")
CROP_DIR: Path = _path("CROP_DIR", "data/plate_crops")
DB_PATH: Path = _path("DB_PATH", "data/anpr.db")

# ── Detection ─────────────────────────────────────────────────────────────────
DETECTOR_CONFIDENCE: float = float(os.getenv("DETECTOR_CONFIDENCE", "0.4"))

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.3"))

# ── Camera ───────────────────────────────────────────────────────────────────
DEFAULT_CAMERA_ID: int = int(os.getenv("DEFAULT_CAMERA_ID", "0"))

# ── Duplicate suppression ─────────────────────────────────────────────────────
# Observations of the same plate+camera within this window are ignored.
DUPLICATE_WINDOW_SEC: int = int(os.getenv("DUPLICATE_WINDOW_SEC", "30"))

# ── Simulated camera IDs for two-camera demo ──────────────────────────────────
CAM_ID_1: str = os.getenv("CAM_ID_1", "CAM001")
CAM_ID_2: str = os.getenv("CAM_ID_2", "CAM002")
