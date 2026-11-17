"""
tests/test_hotlist.py
─────────────────────
Phase 6 — Tests for hotlist matching and alert generation.

Covers:
  • Exact normalized hotlist matching.
  • Case and punctuation tolerance through normalization.
  • Non-matching behavior (no false alerts).
  • Strict rejection of fuzzy matches (differing characters must not match).
  • Alert creation with observation metadata.
  • Duplicate alert suppression within time window.
  • Camera-isolated suppression (same plate on different camera is not suppressed).
  • Alert acknowledgment.
"""

from __future__ import annotations

import pathlib
import pytest

from anpr.alerts import AlertManager
from anpr.database import Database
from anpr.hotlist import HotlistMatcher


@pytest.fixture
def test_db(tmp_path: pathlib.Path) -> Database:
    """Provide an isolated SQLite database."""
    return Database(db_path=tmp_path / "test_hotlist.db")


@pytest.fixture
def matcher(test_db: Database) -> HotlistMatcher:
    """Provide HotlistMatcher connected to test database."""
    return HotlistMatcher(db=test_db)


@pytest.fixture
def alert_manager(test_db: Database) -> AlertManager:
    """Provide AlertManager connected to test database with 30s window."""
    return AlertManager(db=test_db, duplicate_window_sec=30)


# ---------------------------------------------------------------------------
# Hotlist matching tests
# ---------------------------------------------------------------------------


def test_hotlist_exact_match(matcher: HotlistMatcher):
    """Plate matching should succeed when normalized strings match exactly."""
    matcher.add_to_hotlist(
        plate_number="DL 8C AB 1234",
        status="STOLEN",
        description="Silver Sedan stolen from Connaught Place",
    )

    match = matcher.check("dl-8c-ab-1234")
    assert match is not None
    assert match["plate_number"] == "DL8CAB1234"
    assert match["status"] == "STOLEN"
    assert matcher.is_hotlisted("DL 8C AB 1234") is True


def test_hotlist_non_match(matcher: HotlistMatcher):
    """Non-hotlisted plate should return None without error."""
    matcher.add_to_hotlist(plate_number="MH12DE1433", status="WANTED")

    assert matcher.check("KA01E9999") is None
    assert matcher.is_hotlisted("KA01E9999") is False


def test_no_fuzzy_matching(matcher: HotlistMatcher):
    """Fuzzy or near matches must NOT match (to prevent false alerts)."""
    matcher.add_to_hotlist(plate_number="DL8CAB1234", status="STOLEN")

    # One character different at end
    assert matcher.check("DL8CAB1235") is None
    # One letter different
    assert matcher.check("DL8CBB1234") is None
    # Substring
    assert matcher.check("DL8CAB123") is None


def test_hotlist_empty_plate(matcher: HotlistMatcher):
    """Checking None or empty string returns None safely."""
    assert matcher.check(None) is None
    assert matcher.check("") is None
    assert matcher.check("   ") is None


# ---------------------------------------------------------------------------
# Alert generation & Duplicate suppression tests
# ---------------------------------------------------------------------------


def test_alert_created_on_hotlist_match(
    test_db: Database, alert_manager: AlertManager, matcher: HotlistMatcher
):
    """A hotlisted observation should create an alert linked to the observation."""
    matcher.add_to_hotlist(
        plate_number="MH12DE1433",
        status="SUSPECT",
        description="Suspect vehicle in inquiry",
    )
    rec_id = test_db.insert_observation(
        plate_text_raw="MH 12 DE 1433",
        plate_number="MH12DE1433",
        ocr_confidence=0.92,
        camera_id="CAM001",
        image_path="data/plate_crops/crop_mh12.jpg",
        detection_confidence=0.85,
        detected_at="2026-11-17 12:00:00",
    )

    entry = matcher.check("MH12DE1433")
    assert entry is not None

    alert = alert_manager.create_hotlist_alert(
        vehicle_record_id=rec_id,
        plate_number="MH12DE1433",
        camera_id="CAM001",
        hotlist_entry=entry,
        image_path="data/plate_crops/crop_mh12.jpg",
        detected_at="2026-11-17 12:00:00",
    )

    assert alert is not None
    assert alert["plate_number"] == "MH12DE1433"
    assert alert["status"] == "SUSPECT"
    assert alert["camera_id"] == "CAM001"
    assert "SUSPECT" in alert["message"]

    # Verify alert is persisted in SQLite
    alerts_in_db = alert_manager.list_alerts()
    assert len(alerts_in_db) == 1
    assert alerts_in_db[0]["id"] == alert["id"]
    assert alerts_in_db[0]["vehicle_record_id"] == rec_id


def test_duplicate_alert_suppression_consecutive_frames(
    test_db: Database, alert_manager: AlertManager, matcher: HotlistMatcher
):
    """Repeated observations of the same plate within 30s window must NOT create duplicate alerts."""
    matcher.add_to_hotlist(plate_number="KA01E9999", status="STOLEN")

    rec1 = test_db.insert_observation(
        plate_text_raw="KA 01 E 9999",
        plate_number="KA01E9999",
        ocr_confidence=0.95,
        camera_id="CAM001",
        image_path="data/plate_crops/ka_1.jpg",
        detection_confidence=0.88,
        detected_at="2026-11-17 12:00:00",
    )
    entry = matcher.check("KA01E9999")
    assert entry is not None

    # First observation → Alert created
    alert1 = alert_manager.create_hotlist_alert(
        vehicle_record_id=rec1,
        plate_number="KA01E9999",
        camera_id="CAM001",
        hotlist_entry=entry,
        image_path="data/plate_crops/ka_1.jpg",
        detected_at="2026-11-17 12:00:00",
    )
    assert alert1 is not None

    # Consecutive frame 2 seconds later on SAME camera → Alert suppressed
    rec2 = test_db.insert_observation(
        plate_text_raw="KA 01 E 9999",
        plate_number="KA01E9999",
        ocr_confidence=0.94,
        camera_id="CAM001",
        image_path="data/plate_crops/ka_2.jpg",
        detection_confidence=0.89,
        detected_at="2026-11-17 12:00:02",
    )
    alert2 = alert_manager.create_hotlist_alert(
        vehicle_record_id=rec2,
        plate_number="KA01E9999",
        camera_id="CAM001",
        hotlist_entry=entry,
        image_path="data/plate_crops/ka_2.jpg",
        detected_at="2026-11-17 12:00:02",
    )
    assert alert2 is None  # Suppressed!

    # Only 1 alert should exist in DB
    assert len(alert_manager.list_alerts()) == 1


def test_alert_not_suppressed_across_different_cameras(
    test_db: Database, alert_manager: AlertManager, matcher: HotlistMatcher
):
    """Same plate observed on a different camera should trigger an alert for that camera."""
    matcher.add_to_hotlist(plate_number="KA01E9999", status="STOLEN")
    entry = matcher.check("KA01E9999")
    assert entry is not None

    # Camera 1 observation
    rec1 = test_db.insert_observation(
        plate_text_raw="KA 01 E 9999",
        plate_number="KA01E9999",
        ocr_confidence=0.9,
        camera_id="CAM001",
        image_path="data/plate_crops/ka_1.jpg",
        detection_confidence=0.8,
        detected_at="2026-11-17 12:00:00",
    )
    alert1 = alert_manager.create_hotlist_alert(
        vehicle_record_id=rec1,
        plate_number="KA01E9999",
        camera_id="CAM001",
        hotlist_entry=entry,
        image_path="data/plate_crops/ka_1.jpg",
        detected_at="2026-11-17 12:00:00",
    )
    assert alert1 is not None

    # Camera 2 observation 5 seconds later → NOT suppressed because camera differs
    rec2 = test_db.insert_observation(
        plate_text_raw="KA 01 E 9999",
        plate_number="KA01E9999",
        ocr_confidence=0.9,
        camera_id="CAM002",
        image_path="data/plate_crops/ka_2.jpg",
        detection_confidence=0.8,
        detected_at="2026-11-17 12:00:05",
    )
    alert2 = alert_manager.create_hotlist_alert(
        vehicle_record_id=rec2,
        plate_number="KA01E9999",
        camera_id="CAM002",
        hotlist_entry=entry,
        image_path="data/plate_crops/ka_2.jpg",
        detected_at="2026-11-17 12:00:05",
    )
    assert alert2 is not None
    assert alert2["camera_id"] == "CAM002"

    # Both alerts should exist in DB
    assert len(alert_manager.list_alerts()) == 2


def test_alert_triggered_after_cooldown_expires(
    test_db: Database, alert_manager: AlertManager, matcher: HotlistMatcher
):
    """Observation after the duplicate window (35s > 30s) triggers a new alert."""
    matcher.add_to_hotlist(plate_number="KA01E9999", status="STOLEN")
    entry = matcher.check("KA01E9999")
    assert entry is not None

    # Initial alert at 12:00:00
    rec1 = test_db.insert_observation(
        plate_text_raw="KA 01 E 9999",
        plate_number="KA01E9999",
        ocr_confidence=0.9,
        camera_id="CAM001",
        image_path="data/plate_crops/ka_1.jpg",
        detection_confidence=0.8,
        detected_at="2026-11-17 12:00:00",
    )
    alert_manager.create_hotlist_alert(
        vehicle_record_id=rec1,
        plate_number="KA01E9999",
        camera_id="CAM001",
        hotlist_entry=entry,
        image_path="data/plate_crops/ka_1.jpg",
        detected_at="2026-11-17 12:00:00",
    )

    # Observation 45 seconds later (beyond 30s window) → New alert created
    rec2 = test_db.insert_observation(
        plate_text_raw="KA 01 E 9999",
        plate_number="KA01E9999",
        ocr_confidence=0.9,
        camera_id="CAM001",
        image_path="data/plate_crops/ka_2.jpg",
        detection_confidence=0.8,
        detected_at="2026-11-17 12:00:45",
    )
    alert2 = alert_manager.create_hotlist_alert(
        vehicle_record_id=rec2,
        plate_number="KA01E9999",
        camera_id="CAM001",
        hotlist_entry=entry,
        image_path="data/plate_crops/ka_2.jpg",
        detected_at="2026-11-17 12:00:45",
    )
    assert alert2 is not None
    assert len(alert_manager.list_alerts()) == 2


def test_acknowledge_alert(
    test_db: Database, alert_manager: AlertManager, matcher: HotlistMatcher
):
    """Alert acknowledgment works correctly through AlertManager."""
    matcher.add_to_hotlist(plate_number="UP32AB0001", status="WANTED")
    entry = matcher.check("UP32AB0001")
    assert entry is not None

    rec = test_db.insert_observation(
        plate_text_raw="UP32AB0001",
        plate_number="UP32AB0001",
        ocr_confidence=0.9,
        camera_id="CAM001",
        image_path="data/plate_crops/up.jpg",
        detection_confidence=0.8,
    )
    alert = alert_manager.create_hotlist_alert(
        vehicle_record_id=rec,
        plate_number="UP32AB0001",
        camera_id="CAM001",
        hotlist_entry=entry,
        image_path="data/plate_crops/up.jpg",
    )
    assert alert is not None

    # Verify unacknowledged
    assert len(alert_manager.list_alerts(acknowledged=False)) == 1

    # Acknowledge
    assert alert_manager.acknowledge_alert(alert["id"]) is True

    # Verify acknowledged
    assert len(alert_manager.list_alerts(acknowledged=False)) == 0
    assert len(alert_manager.list_alerts(acknowledged=True)) == 1
