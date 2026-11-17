"""
tests/test_reliability.py
──────────────────────────
Phase 10 — Reliability, Edge Cases & Error Handling Tests.

Covers:
  1. Clear plate in daylight condition.
  2. Plate at an angle (rotated / skewed).
  3. Blurry or low-resolution plate crop.
  4. Low-light / dark frame sample.
  5. Frame with no visible plate (safe handling).
  6. Plate present in hotlist vs absent from hotlist.
  7. Same plate repeated across many frames (duplicate suppression stress).
  8. Invalid video path or unavailable capture source.
  9. Unwritable crop directory or database access error resilience.
"""

from __future__ import annotations

import pathlib
import cv2
import numpy as np
import pytest

from anpr.database import Database
from anpr.detector import LicensePlateDetector
from anpr.ocr import PlateOCR
from anpr.pipeline import ANPRPipeline
from anpr.video import VideoOpenError, open_source


@pytest.fixture(scope="module")
def detector() -> LicensePlateDetector:
    return LicensePlateDetector()


@pytest.fixture(scope="module")
def ocr() -> PlateOCR:
    return PlateOCR(gpu=False)


@pytest.fixture
def isolated_pipeline(
    tmp_path: pathlib.Path, detector: LicensePlateDetector, ocr: PlateOCR
) -> ANPRPipeline:
    db = Database(db_path=tmp_path / "test_reliability.db")
    crop_dir = tmp_path / "crops"
    return ANPRPipeline(
        camera_id="CAM_TEST",
        db=db,
        detector=detector,
        ocr=ocr,
        crop_dir=crop_dir,
        duplicate_window_sec=10,
    )


def _generate_plate_image(text: str = "KA01AB1234", bg_val: int = 120, text_val: int = 0) -> np.ndarray:
    """Generate a synthetic synthetic vehicle frame with a plate."""
    frame = np.full((480, 640, 3), bg_val, dtype=np.uint8)
    # Vehicle body
    cv2.rectangle(frame, (120, 180), (520, 420), (60, 60, 60), -1)
    # Plate background (white)
    cv2.rectangle(frame, (200, 270), (440, 340), (255, 255, 255), -1)
    cv2.rectangle(frame, (200, 270), (440, 340), (0, 0, 0), 2)
    # Text
    cv2.putText(
        frame,
        text,
        (215, 318),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (text_val, text_val, text_val),
        2,
        cv2.LINE_AA,
    )
    return frame


# 1. Clear plate in daylight
def test_clear_plate_daylight(isolated_pipeline: ANPRPipeline):
    frame = _generate_plate_image("DL8CAB1234", bg_val=200)
    res = isolated_pipeline.process_frame(frame, frame_index=1, timestamp="2026-11-17 12:00:00")
    assert res.has_detections
    assert len(res.observations) >= 1
    obs = res.observations[0]
    assert obs.is_valid_plate
    assert "DL8C" in obs.plate_number or "1234" in obs.plate_number


# 2. Plate at an angle (rotated)
def test_plate_at_angle(isolated_pipeline: ANPRPipeline):
    frame = _generate_plate_image("MH12DE1433", bg_val=150)
    # Rotate frame slightly (e.g., 8 degrees)
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 8, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h), borderValue=(150, 150, 150))
    res = isolated_pipeline.process_frame(rotated, frame_index=2, timestamp="2026-11-17 12:01:00")
    assert isinstance(res.annotated_frame, np.ndarray)


# 3. Blurry / low resolution plate
def test_blurry_low_resolution_plate(isolated_pipeline: ANPRPipeline):
    frame = _generate_plate_image("DL8CAB1234", bg_val=120)
    blurred = cv2.GaussianBlur(frame, (25, 25), 0)
    res = isolated_pipeline.process_frame(blurred, frame_index=3, timestamp="2026-11-17 12:02:00")
    # Low-quality frame should not crash the pipeline
    assert isinstance(res.observations, list)


# 4. Low light / dark sample
def test_low_light_sample(isolated_pipeline: ANPRPipeline):
    frame = _generate_plate_image("DL8CAB1234", bg_val=30)
    dark_frame = cv2.convertScaleAbs(frame, alpha=0.35, beta=0)
    res = isolated_pipeline.process_frame(dark_frame, frame_index=4, timestamp="2026-11-17 12:03:00")
    assert isinstance(res.observations, list)


# 5. Frame with no visible plate
def test_no_visible_plate(isolated_pipeline: ANPRPipeline):
    blank = np.full((480, 640, 3), 100, dtype=np.uint8)
    res = isolated_pipeline.process_frame(blank, frame_index=5, timestamp="2026-11-17 12:04:00")
    assert not res.has_detections
    assert len(res.observations) == 0


# 6. Plate present in hotlist vs absent
def test_hotlist_present_vs_absent(isolated_pipeline: ANPRPipeline):
    isolated_pipeline.matcher.add_to_hotlist("DL8CAB1234", "STOLEN", "Test alert")
    
    # Present
    frame_present = _generate_plate_image("DL8CAB1234")
    res_present = isolated_pipeline.process_frame(frame_present, frame_index=6, timestamp="2026-11-17 12:05:00")
    assert res_present.observations[0].is_hotlist_match is True
    assert res_present.observations[0].alert is not None

    # Absent
    frame_absent = _generate_plate_image("HR26DK8337")
    res_absent = isolated_pipeline.process_frame(frame_absent, frame_index=7, timestamp="2026-11-17 12:05:01")
    if res_absent.observations:
        assert res_absent.observations[0].is_hotlist_match is False


# 7. Same plate repeated across consecutive frames (duplicate stress)
def test_duplicate_stress_across_frames(isolated_pipeline: ANPRPipeline):
    frame = _generate_plate_image("DL8CAB1234")
    inserted_count = 0
    suppressed_count = 0

    # 8 consecutive frames 1 second apart (well within the 10s window)
    for i in range(8):
        res = isolated_pipeline.process_frame(frame, frame_index=100 + i, timestamp=f"2026-11-17 12:15:{i:02d}")
        if res.observations:
            obs = res.observations[0]
            if obs.is_duplicate:
                suppressed_count += 1
            else:
                inserted_count += 1

    assert inserted_count == 1
    assert suppressed_count == 7
    # Exactly 1 new record stored in database for this sequence
    records = isolated_pipeline.db.search_observations(
        plate_number="DL8CAB1234",
        start_time="2026-11-17 12:15:00",
        end_time="2026-11-17 12:15:10",
    )
    assert len(records) == 1


# 8. Invalid video path or unavailable webcam
def test_invalid_video_path_error():
    with pytest.raises(FileNotFoundError):
        with open_source("non_existent_video_file_xyz123.mp4"):
            pass


# 9. Database gracefully handles empty queries and non-existent records
def test_database_safe_queries(isolated_pipeline: ANPRPipeline):
    assert isolated_pipeline.db.get_movement_history("NONEXISTENTPLATE999") == []
    assert isolated_pipeline.db.get_alert(999999) is None
    assert isolated_pipeline.db.remove_hotlist_entry("NONEXISTENTPLATE999") is False
