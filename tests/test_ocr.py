"""
tests/test_ocr.py
─────────────────
Phase 4 — Tests for anpr/ocr.py.

Covers:
  • Reader initialization and caching.
  • Safe handling of empty, invalid, or blank images.
  • Optical character recognition on clear synthetic plate crop.
  • Reading from file path vs numpy array.
  • Confidence score calculation and cutoff filtering.
  • Interoperability with plate normalization and validation.
"""

from __future__ import annotations

import pathlib
import cv2
import numpy as np
import pytest

from anpr.ocr import OCRResult, PlateOCR
from anpr.plate_utils import is_valid_indian_plate, normalize_plate, validate_plate


@pytest.fixture(scope="module")
def ocr_reader() -> PlateOCR:
    """Instantiate the OCR engine once for the module."""
    return PlateOCR(gpu=False)


@pytest.fixture
def clean_plate_crop() -> np.ndarray:
    """Create a high-contrast synthetic plate crop."""
    crop = np.full((70, 260, 3), 255, dtype=np.uint8)
    # Draw dark registration text
    cv2.putText(
        crop,
        "DL 8C AB 1234",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return crop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ocr_reader_initialized(ocr_reader: PlateOCR):
    """PlateOCR reader instance should be loaded and have an internal easyocr reader."""
    assert ocr_reader.reader is not None


def test_ocr_read_empty_or_none(ocr_reader: PlateOCR):
    """Empty or None image input must return an empty OCRResult safely."""
    res1 = ocr_reader.read_plate(None)  # type: ignore
    assert isinstance(res1, OCRResult)
    assert res1.is_empty
    assert res1.raw_text == ""
    assert res1.confidence == 0.0

    res2 = ocr_reader.read_plate(np.array([], dtype=np.uint8))
    assert res2.is_empty


def test_ocr_read_blank_image(ocr_reader: PlateOCR):
    """Blank image with no characters should return empty OCRResult."""
    blank = np.full((100, 200, 3), 200, dtype=np.uint8)
    res = ocr_reader.read_plate(blank)
    assert res.is_empty
    assert res.confidence == 0.0


def test_ocr_read_clear_plate(ocr_reader: PlateOCR, clean_plate_crop: np.ndarray):
    """Clear plate image must return recognizable text and positive confidence."""
    res = ocr_reader.read_plate(clean_plate_crop)

    assert not res.is_empty
    assert res.confidence > 0.4
    assert len(res.tokens) > 0

    # Normalization should yield DL8CAB1234
    normalized = normalize_plate(res.raw_text)
    assert "DL" in normalized
    assert "1234" in normalized


def test_ocr_read_from_file_path(
    ocr_reader: PlateOCR, clean_plate_crop: np.ndarray, tmp_path: pathlib.Path
):
    """read_plate should work seamlessly with a Path or str file path."""
    img_path = tmp_path / "test_crop.jpg"
    cv2.imwrite(str(img_path), clean_plate_crop)

    res = ocr_reader.read_plate(img_path)
    assert not res.is_empty
    assert res.confidence > 0.4


def test_ocr_missing_file_path(ocr_reader: PlateOCR):
    """Non-existent file path returns an empty OCRResult without raising an unhandled exception."""
    res = ocr_reader.read_plate("/tmp/definitely_not_a_real_crop_xyz123.jpg")
    assert res.is_empty


def test_ocr_high_confidence_cutoff(
    ocr_reader: PlateOCR, clean_plate_crop: np.ndarray
):
    """Extremely high cutoff filters all tokens and yields an empty result."""
    res = ocr_reader.read_plate(clean_plate_crop, min_confidence=0.9999)
    assert res.is_empty


def test_ocr_and_plate_validation_pipeline(
    ocr_reader: PlateOCR, clean_plate_crop: np.ndarray
):
    """End-to-end Phase 4 pipeline: OCR -> normalize -> validate."""
    ocr_res = ocr_reader.read_plate(clean_plate_crop)
    assert not ocr_res.is_empty

    raw = ocr_res.raw_text
    normalized = normalize_plate(raw)
    assert normalized == "DL8CAB1234"

    # Both raw and normalized values remain distinguishable
    assert raw != normalized

    # Validation check
    val = validate_plate(raw)
    assert val["is_valid"] is True
    assert val["normalized"] == "DL8CAB1234"
    assert val["state_code"] == "DL"
    assert is_valid_indian_plate(raw) is True
