"""
tests/test_detector.py
──────────────────────
Phase 3 — Tests for anpr/detector.py.

Covers:
  • Detector initialization and FileNotFoundError for missing model weights.
  • Configurable confidence threshold.
  • Safe handling of empty/corrupt frames.
  • Zero detections on frames without license plates.
  • Valid detection of a visible license plate with plausible bounding box.
  • Plate cropping, boundary clamping, and invalid crop rejection.
  • Saving crops to disk and verifying saved image file.
  • Drawing bounding box annotations on frame copies.
"""

from __future__ import annotations

import pathlib
import cv2
import numpy as np
import pytest

from anpr.detector import Detection, LicensePlateDetector


@pytest.fixture(scope="module")
def detector() -> LicensePlateDetector:
    """Fixture to instantiate the detector once for test suite."""
    return LicensePlateDetector()


@pytest.fixture
def sample_plate_image() -> np.ndarray:
    """Create a synthetic vehicle image with a visible license plate."""
    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Vehicle bumper
    cv2.rectangle(img, (150, 200), (490, 400), (50, 50, 50), -1)
    # License plate (white rectangle with black border and alphanumeric text)
    cv2.rectangle(img, (220, 280), (420, 340), (255, 255, 255), -1)
    cv2.rectangle(img, (220, 280), (420, 340), (0, 0, 0), 2)
    cv2.putText(
        img,
        "DL 8C AB 1234",
        (230, 325),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    return img


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


def test_detector_missing_model_raises_error():
    """Initializing detector with non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        LicensePlateDetector(model_path="/nonexistent/model.pt")


def test_detector_loads_successfully(detector: LicensePlateDetector):
    """Detector should load YOLO model weights and report class names."""
    assert detector.model is not None
    assert 0 in detector.model.names
    assert detector.model.names[0] == "license_plate"


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


def test_detect_empty_or_none_frame(detector: LicensePlateDetector):
    """detect() should return an empty list safely for None or empty arrays."""
    assert detector.detect(None) == []  # type: ignore
    assert detector.detect(np.array([], dtype=np.uint8)) == []


def test_detect_no_plate_frame(detector: LicensePlateDetector):
    """A plain black frame with no plate must return zero detections."""
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(black_frame)
    assert len(detections) == 0


def test_detect_visible_plate(
    detector: LicensePlateDetector, sample_plate_image: np.ndarray
):
    """Visible plate frame should return a plausible detection bounding box."""
    detections = detector.detect(sample_plate_image)
    assert len(detections) >= 1

    det = detections[0]
    assert isinstance(det, Detection)
    assert det.is_valid()
    assert det.confidence >= 0.4
    assert det.class_name == "license_plate"

    # Plate was drawn at (220, 280, 420, 340). Bounding box should overlap closely.
    assert 180 <= det.x1 <= 240
    assert 250 <= det.y1 <= 300
    assert 400 <= det.x2 <= 460
    assert 320 <= det.y2 <= 380


def test_configurable_threshold(
    detector: LicensePlateDetector, sample_plate_image: np.ndarray
):
    """Configuring a higher threshold than model confidence should filter detections."""
    # Threshold 0.999 should return no detections
    high_thresh_dets = detector.detect(sample_plate_image, confidence_threshold=0.999)
    assert len(high_thresh_dets) == 0

    # Low threshold should return detections
    low_thresh_dets = detector.detect(sample_plate_image, confidence_threshold=0.2)
    assert len(low_thresh_dets) >= 1


# ---------------------------------------------------------------------------
# Cropping tests
# ---------------------------------------------------------------------------


def test_crop_plate_valid(
    detector: LicensePlateDetector, sample_plate_image: np.ndarray
):
    """crop_plate() should extract the bounded sub-image correctly."""
    detections = detector.detect(sample_plate_image)
    assert len(detections) >= 1

    crop = detector.crop_plate(sample_plate_image, detections[0])
    assert crop is not None
    assert isinstance(crop, np.ndarray)
    assert crop.ndim == 3
    assert crop.shape[0] == detections[0].height
    assert crop.shape[1] == detections[0].width


def test_crop_plate_invalid_bounds(detector: LicensePlateDetector):
    """crop_plate() should return None when detection coordinates are invalid."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    invalid_det = Detection(x1=50, y1=50, x2=50, y2=50, confidence=0.9)
    assert detector.crop_plate(frame, invalid_det) is None

    out_of_bounds_det = Detection(x1=200, y1=200, x2=300, y2=300, confidence=0.9)
    assert detector.crop_plate(frame, out_of_bounds_det) is None


# ---------------------------------------------------------------------------
# Saving and drawing tests
# ---------------------------------------------------------------------------


def test_save_crop(tmp_path: pathlib.Path):
    """save_crop() should write image to disk and return valid Path."""
    dummy_crop = np.full((50, 150, 3), 255, dtype=np.uint8)
    saved_path = LicensePlateDetector.save_crop(dummy_crop, output_dir=tmp_path)

    assert saved_path.exists()
    assert saved_path.is_file()
    assert saved_path.suffix == ".jpg"

    loaded = cv2.imread(str(saved_path))
    assert loaded is not None
    assert loaded.shape == (50, 150, 3)


def test_save_empty_crop_raises_value_error():
    """Saving an empty or None crop must raise ValueError."""
    with pytest.raises(ValueError, match="empty or invalid"):
        LicensePlateDetector.save_crop(np.array([], dtype=np.uint8))


def test_draw_detections(sample_plate_image: np.ndarray):
    """draw_detections() returns annotated copy without mutating original when copy=True."""
    orig = sample_plate_image.copy()
    detections = [
        Detection(x1=10, y1=10, x2=100, y2=50, confidence=0.85, class_name="license_plate")
    ]
    annotated = LicensePlateDetector.draw_detections(sample_plate_image, detections, copy=True)

    assert annotated.shape == orig.shape
    # Annotated image should differ from original
    assert not np.array_equal(annotated, orig)
    # Original should remain untouched
    assert np.array_equal(sample_plate_image, orig)
