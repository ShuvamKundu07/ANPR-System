"""
tests/test_tracking.py
───────────────────────
Phase 9 — Two-Camera Movement History & Tracking Tests.

Covers:
  • Two cameras (e.g. CAM001, CAM002) observing the same plate number.
  • Staggered timestamps producing an ordered chronological journey.
  • VehicleJourneyTracker and Database.get_movement_history verification.
  • Ensuring different plates are kept distinct and not merged.
  • Duplicate suppression working per-camera without blocking other cameras.
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timedelta, timezone

import pytest

from anpr.database import Database
from anpr.tracking import DuplicateTracker, VehicleJourneyTracker


@pytest.fixture
def test_db(tmp_path: pathlib.Path) -> Database:
    """Provide an isolated SQLite database for tracking tests."""
    return Database(db_path=tmp_path / "test_tracking.db")


def test_two_camera_ordered_movement_history(test_db: Database) -> None:
    """The same plate observed at CAM001 and later CAM002 appears in chronological order."""
    plate = "DL8CAB1234"
    time_cam1 = "2026-11-17 10:30:00"
    time_cam2 = "2026-11-17 10:42:00"

    # Insert observation from CAM001 at 10:30
    id1 = test_db.insert_observation(
        plate_text_raw="DL 8C AB 1234",
        plate_number=plate,
        ocr_confidence=0.92,
        camera_id="CAM001",
        image_path="data/plate_crops/crop_cam1.jpg",
        detection_confidence=0.88,
        detected_at=time_cam1,
    )

    # Insert observation from CAM002 at 10:42
    id2 = test_db.insert_observation(
        plate_text_raw="DL-8C-AB-1234",
        plate_number=plate,
        ocr_confidence=0.89,
        camera_id="CAM002",
        image_path="data/plate_crops/crop_cam2.jpg",
        detection_confidence=0.85,
        detected_at=time_cam2,
    )

    tracker = VehicleJourneyTracker(db=test_db)
    journey = tracker.get_journey(plate)

    assert len(journey) == 2
    # Verify chronological ordering
    assert journey[0]["id"] == id1
    assert journey[0]["camera_id"] == "CAM001"
    assert journey[0]["detected_at"] == time_cam1

    assert journey[1]["id"] == id2
    assert journey[1]["camera_id"] == "CAM002"
    assert journey[1]["detected_at"] == time_cam2


def test_movement_history_does_not_merge_different_plates(test_db: Database) -> None:
    """Observations of different plates must not be merged in movement history."""
    test_db.insert_observation(
        plate_text_raw="MH 12 AB 1234",
        plate_number="MH12AB1234",
        ocr_confidence=0.95,
        camera_id="CAM001",
        image_path="crop1.jpg",
        detection_confidence=0.90,
        detected_at="2026-11-17 11:00:00",
    )
    test_db.insert_observation(
        plate_text_raw="KA 01 CD 5678",
        plate_number="KA01CD5678",
        ocr_confidence=0.91,
        camera_id="CAM002",
        image_path="crop2.jpg",
        detection_confidence=0.87,
        detected_at="2026-11-17 11:15:00",
    )

    tracker = VehicleJourneyTracker(db=test_db)
    mh_journey = tracker.get_journey("MH12AB1234")
    ka_journey = tracker.get_journey("KA01CD5678")

    assert len(mh_journey) == 1
    assert mh_journey[0]["plate_number"] == "MH12AB1234"
    assert mh_journey[0]["camera_id"] == "CAM001"

    assert len(ka_journey) == 1
    assert ka_journey[0]["plate_number"] == "KA01CD5678"
    assert ka_journey[0]["camera_id"] == "CAM002"


def test_duplicate_suppression_per_camera(test_db: Database) -> None:
    """Duplicate suppression on CAM001 does not prevent immediate observation on CAM002."""
    tracker = DuplicateTracker(db=test_db, window_seconds=30)
    plate = "UP16AB9999"
    t0 = "2026-11-17 12:00:00"
    t1 = "2026-11-17 12:00:05"

    # Observe on CAM001
    assert tracker.is_duplicate("CAM001", plate, timestamp=t0) is False
    tracker.record_observation("CAM001", plate, timestamp=t0)

    # 5 seconds later on CAM001 -> Duplicate
    assert tracker.is_duplicate("CAM001", plate, timestamp=t1) is True

    # 5 seconds later on CAM002 -> NOT Duplicate (different camera)
    assert tracker.is_duplicate("CAM002", plate, timestamp=t1) is False
