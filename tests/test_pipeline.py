"""
tests/test_pipeline.py
───────────────────────
Phase 7 — End-to-End ANPR Pipeline Tests.

Covers:
  • Full pipeline initialization.
  • Processing frames without plates (safe no-op).
  • Processing frame with a visible license plate (detection -> crop -> OCR -> DB).
  • Plate bounding box and text annotation rendering.
  • Consecutive frame duplicate suppression (no database flooding).
  • Hotlist matching and alert dispatch through the pipeline.
  • Multi-frame video processing through process_video().
"""

from __future__ import annotations

import pathlib
import cv2
import numpy as np
import pytest

from anpr.database import Database
from anpr.detector import LicensePlateDetector
from anpr.ocr import PlateOCR
from anpr.pipeline import ANPRPipeline, PipelineResult


@pytest.fixture(scope="module")
def shared_detector() -> LicensePlateDetector:
    return LicensePlateDetector()


@pytest.fixture(scope="module")
def shared_ocr() -> PlateOCR:
    return PlateOCR(gpu=False)


@pytest.fixture
def test_pipeline(
    tmp_path: pathlib.Path,
    shared_detector: LicensePlateDetector,
    shared_ocr: PlateOCR,
) -> ANPRPipeline:
    """Create a pipeline using isolated DB and crop storage, reusing heavy models."""
    db = Database(db_path=tmp_path / "test_pipeline.db")
    crop_dir = tmp_path / "plate_crops"
    return ANPRPipeline(
        camera_id="CAM001",
        db=db,
        detector=shared_detector,
        ocr=shared_ocr,
        crop_dir=crop_dir,
        duplicate_window_sec=30,
    )


@pytest.fixture
def vehicle_plate_frame() -> np.ndarray:
    """Generate a test frame containing a vehicle and license plate."""
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame, (150, 200), (490, 400), (50, 50, 50), -1)
    # License plate rectangle with sufficient margin
    cv2.rectangle(frame, (210, 275), (430, 345), (255, 255, 255), -1)
    cv2.rectangle(frame, (210, 275), (430, 345), (0, 0, 0), 2)
    cv2.putText(
        frame,
        "DL 8C AB 1234",
        (220, 322),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------


def test_pipeline_initialization(test_pipeline: ANPRPipeline):
    """Pipeline components should be initialized and linked."""
    assert test_pipeline.camera_id == "CAM001"
    assert test_pipeline.detector is not None
    assert test_pipeline.ocr is not None
    assert test_pipeline.db is not None
    assert test_pipeline.matcher is not None
    assert test_pipeline.alert_manager is not None
    assert test_pipeline.duplicate_tracker is not None


def test_pipeline_process_empty_frame(test_pipeline: ANPRPipeline):
    """Processing a black frame with no plates returns a safe empty result."""
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = test_pipeline.process_frame(black_frame, frame_index=0)

    assert isinstance(res, PipelineResult)
    assert res.frame_index == 0
    assert not res.has_detections
    assert len(res.observations) == 0
    # Annotated frame should match input dimensions
    assert res.annotated_frame.shape == (480, 640, 3)


def test_pipeline_process_visible_plate(
    test_pipeline: ANPRPipeline, vehicle_plate_frame: np.ndarray
):
    """Full pipeline execution on a frame with a visible plate."""
    res = test_pipeline.process_frame(
        vehicle_plate_frame, frame_index=1, timestamp="2026-11-17 14:00:00"
    )

    assert res.has_detections
    assert len(res.observations) >= 1

    obs = res.observations[0]
    assert obs.is_valid_plate
    assert "DL" in obs.plate_number
    assert obs.db_record_id is not None
    assert obs.db_record_id >= 1
    assert not obs.is_duplicate

    # Verify saved crop image exists
    assert obs.image_path is not None
    assert obs.image_path.exists()

    # Verify observation in SQLite database
    db_records = test_pipeline.db.get_recent_observations()
    assert len(db_records) == 1
    assert db_records[0]["id"] == obs.db_record_id
    assert db_records[0]["camera_id"] == "CAM001"

    # Verify annotated frame was drawn upon
    assert not np.array_equal(res.annotated_frame, vehicle_plate_frame)


def test_pipeline_duplicate_suppression(
    test_pipeline: ANPRPipeline, vehicle_plate_frame: np.ndarray
):
    """Consecutive frames of the same vehicle should not flood the database."""
    # Frame 1: Initial observation
    res1 = test_pipeline.process_frame(
        vehicle_plate_frame, frame_index=10, timestamp="2026-11-17 14:00:00"
    )
    assert len(res1.observations) >= 1
    obs1 = res1.observations[0]
    assert obs1.db_record_id is not None
    assert not obs1.is_duplicate

    # Frame 2: Immediately following frame (1 second later)
    res2 = test_pipeline.process_frame(
        vehicle_plate_frame, frame_index=11, timestamp="2026-11-17 14:00:01"
    )
    assert len(res2.observations) >= 1
    obs2 = res2.observations[0]
    assert obs2.is_duplicate is True
    assert obs2.db_record_id is None  # Suppressed, no DB write

    # SQLite must only contain 1 record
    records = test_pipeline.db.get_recent_observations()
    assert len(records) == 1


def test_pipeline_hotlist_detection_and_alert(
    test_pipeline: ANPRPipeline, vehicle_plate_frame: np.ndarray
):
    """Hotlist match through pipeline creates alert in SQLite and marks observation."""
    # Pre-seed hotlist with the plate number
    test_pipeline.matcher.add_to_hotlist(
        plate_number="DL 8C AB 1234",
        status="STOLEN",
        description="Stolen car demonstration",
    )

    res = test_pipeline.process_frame(
        vehicle_plate_frame, frame_index=20, timestamp="2026-11-17 14:00:00"
    )

    assert len(res.observations) >= 1
    obs = res.observations[0]
    assert obs.is_hotlist_match is True
    assert obs.alert is not None
    assert obs.alert["status"] == "STOLEN"

    # Verify alert in SQLite
    alerts = test_pipeline.alert_manager.list_alerts()
    assert len(alerts) == 1
    assert alerts[0]["plate_number"] == obs.plate_number
    assert alerts[0]["alert_type"] == "HOTLIST_MATCH"


def test_pipeline_process_video_stream(
    test_pipeline: ANPRPipeline, tmp_path: pathlib.Path
):
    """process_video iterates over frames and returns PipelineResult stream."""
    video_path = tmp_path / "test_mini.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for i in range(5):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    results = list(test_pipeline.process_video(video_path, max_frames=3))
    assert len(results) == 3
    for r in results:
        assert isinstance(r, PipelineResult)
        assert r.annotated_frame.shape == (240, 320, 3)
