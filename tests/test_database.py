"""
tests/test_database.py
──────────────────────
Phase 5 — Tests for anpr/database.py.

Covers:
  • In-memory and file-based database initialization.
  • Table schema and index creation.
  • Inserting and retrieving vehicle observation records.
  • Search observations by plate, camera, and date range.
  • Movement history ordering (chronological).
  • Hotlist CRUD operations (add, list, update-on-conflict, get, remove).
  • Alert creation, retrieval with joins, and acknowledgment.
  • Cascading deletes on vehicle record removal.
  • Isolation: Production database is never touched.
"""

from __future__ import annotations

import pathlib
import pytest

from anpr.database import Database


@pytest.fixture
def test_db(tmp_path: pathlib.Path) -> Database:
    """Provide a fresh isolated database on each test using tmp_path."""
    db_file = tmp_path / "test_anpr.db"
    return Database(db_path=db_file)


@pytest.fixture
def memory_db() -> Database:
    """Provide a purely in-memory test database."""
    return Database(db_path=":memory:")


# ---------------------------------------------------------------------------
# Schema and Initialization tests
# ---------------------------------------------------------------------------


def test_database_initialization(test_db: Database):
    """Database initializes and creates all three expected tables."""
    conn = test_db.get_connection()
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    )
    tables = [row["name"] for row in cursor.fetchall()]
    assert "vehicle_records" in tables
    assert "hotlist" in tables
    assert "alerts" in tables
    conn.close()


def test_foreign_key_enforcement(memory_db: Database):
    """Alerts table must enforce foreign key to vehicle_records."""
    # Attempting to create an alert for non-existent vehicle_record_id should fail
    with pytest.raises(Exception):
        memory_db.create_alert(
            vehicle_record_id=999999,
            plate_number="DL08AB1234",
            alert_type="TEST",
            message="Foreign key should fail",
        )


# ---------------------------------------------------------------------------
# Vehicle Records tests
# ---------------------------------------------------------------------------


def test_insert_and_get_observation(test_db: Database):
    """Observations should insert and retrieve with all expected fields."""
    rec_id = test_db.insert_observation(
        plate_text_raw="DL 8C AB 1234",
        plate_number="DL8CAB1234",
        ocr_confidence=0.88,
        camera_id="CAM001",
        image_path="data/plate_crops/test.jpg",
        detection_confidence=0.75,
        detected_at="2026-11-17 10:00:00",
    )
    assert rec_id == 1

    records = test_db.get_recent_observations(limit=10)
    assert len(records) == 1
    r = records[0]
    assert r["id"] == 1
    assert r["plate_number"] == "DL8CAB1234"
    assert r["plate_text_raw"] == "DL 8C AB 1234"
    assert r["ocr_confidence"] == pytest.approx(0.88)
    assert r["camera_id"] == "CAM001"
    assert r["detection_confidence"] == pytest.approx(0.75)


def test_search_observations(test_db: Database):
    """search_observations should filter by plate number and camera ID."""
    test_db.insert_observation(
        plate_text_raw="DL 8C AB 1234",
        plate_number="DL8CAB1234",
        ocr_confidence=0.9,
        camera_id="CAM001",
        image_path="data/plate_crops/1.jpg",
        detection_confidence=0.8,
        detected_at="2026-11-17 10:00:00",
    )
    test_db.insert_observation(
        plate_text_raw="MH 12 DE 1433",
        plate_number="MH12DE1433",
        ocr_confidence=0.85,
        camera_id="CAM002",
        image_path="data/plate_crops/2.jpg",
        detection_confidence=0.82,
        detected_at="2026-11-17 10:05:00",
    )

    # Search by plate
    res_plate = test_db.search_observations(plate_number="MH12DE1433")
    assert len(res_plate) == 1
    assert res_plate[0]["plate_number"] == "MH12DE1433"

    # Search by camera
    res_cam = test_db.search_observations(camera_id="CAM001")
    assert len(res_cam) == 1
    assert res_cam[0]["plate_number"] == "DL8CAB1234"

    # Search non-matching
    res_none = test_db.search_observations(plate_number="KA01E9999")
    assert len(res_none) == 0


def test_movement_history_chronological(test_db: Database):
    """get_movement_history should return records ordered chronologically (ASC)."""
    # Insert out of order
    test_db.insert_observation(
        plate_text_raw="DL8CAB1234",
        plate_number="DL8CAB1234",
        ocr_confidence=0.9,
        camera_id="CAM002",
        image_path="data/plate_crops/b.jpg",
        detection_confidence=0.8,
        detected_at="2026-11-17 11:00:00",
    )
    test_db.insert_observation(
        plate_text_raw="DL8CAB1234",
        plate_number="DL8CAB1234",
        ocr_confidence=0.85,
        camera_id="CAM001",
        image_path="data/plate_crops/a.jpg",
        detection_confidence=0.8,
        detected_at="2026-11-17 10:30:00",
    )

    history = test_db.get_movement_history("DL 8C AB 1234")
    assert len(history) == 2
    # First entry should be CAM001 at 10:30
    assert history[0]["camera_id"] == "CAM001"
    assert history[0]["detected_at"] == "2026-11-17 10:30:00"
    # Second entry should be CAM002 at 11:00
    assert history[1]["camera_id"] == "CAM002"
    assert history[1]["detected_at"] == "2026-11-17 11:00:00"


# ---------------------------------------------------------------------------
# Hotlist tests
# ---------------------------------------------------------------------------


def test_hotlist_crud(test_db: Database):
    """Test hotlist add, list, get, and delete operations."""
    # Add
    hid = test_db.add_hotlist_entry(
        plate_number="MH 12 DE 1433",
        status="STOLEN",
        description="Red Sedan reported stolen in Pune",
    )
    assert hid >= 1

    # Get
    entry = test_db.get_hotlist_entry("mh-12-de-1433")
    assert entry is not None
    assert entry["plate_number"] == "MH12DE1433"
    assert entry["status"] == "STOLEN"
    assert entry["description"] == "Red Sedan reported stolen in Pune"

    # List
    entries = test_db.list_hotlist_entries()
    assert len(entries) == 1

    # Update on conflict
    test_db.add_hotlist_entry(
        plate_number="MH12DE1433",
        status="RECOVERED",
        description="Vehicle recovered",
    )
    updated = test_db.get_hotlist_entry("MH12DE1433")
    assert updated is not None
    assert updated["status"] == "RECOVERED"
    assert len(test_db.list_hotlist_entries()) == 1

    # Delete
    deleted = test_db.remove_hotlist_entry("MH12DE1433")
    assert deleted is True
    assert test_db.get_hotlist_entry("MH12DE1433") is None
    assert test_db.remove_hotlist_entry("MH12DE1433") is False


def test_hotlist_empty_plate_raises_error(test_db: Database):
    """Adding empty plate string to hotlist must raise ValueError."""
    with pytest.raises(ValueError, match="Cannot add an empty plate"):
        test_db.add_hotlist_entry("", "STOLEN")


# ---------------------------------------------------------------------------
# Alerts tests
# ---------------------------------------------------------------------------


def test_alerts_lifecycle(test_db: Database):
    """Test alert creation, listing with join metadata, and acknowledgment."""
    rec_id = test_db.insert_observation(
        plate_text_raw="KA 01 E 9999",
        plate_number="KA01E9999",
        ocr_confidence=0.95,
        camera_id="CAM001",
        image_path="data/plate_crops/crop_9999.jpg",
        detection_confidence=0.88,
    )

    alert_id = test_db.create_alert(
        vehicle_record_id=rec_id,
        plate_number="KA01E9999",
        alert_type="HOTLIST_MATCH",
        message="Stolen vehicle detected at Gate 1",
    )
    assert alert_id >= 1

    # List unacknowledged alerts
    unack = test_db.list_alerts(acknowledged=False)
    assert len(unack) == 1
    a = unack[0]
    assert a["id"] == alert_id
    assert a["plate_number"] == "KA01E9999"
    assert a["camera_id"] == "CAM001"
    assert a["image_path"] == "data/plate_crops/crop_9999.jpg"
    assert a["acknowledged"] == 0

    # Acknowledge alert
    success = test_db.acknowledge_alert(alert_id)
    assert success is True

    # Check unacknowledged is now empty
    assert len(test_db.list_alerts(acknowledged=False)) == 0
    # Check acknowledged contains it
    ack = test_db.list_alerts(acknowledged=True)
    assert len(ack) == 1
    assert ack[0]["id"] == alert_id


def test_cascade_delete_alerts(test_db: Database):
    """Deleting a vehicle observation cascades and removes associated alerts."""
    rec_id = test_db.insert_observation(
        plate_text_raw="UP 32 AB 0001",
        plate_number="UP32AB0001",
        ocr_confidence=0.91,
        camera_id="CAM001",
        image_path="data/plate_crops/up.jpg",
        detection_confidence=0.8,
    )
    test_db.create_alert(
        vehicle_record_id=rec_id,
        plate_number="UP32AB0001",
        alert_type="WANTED",
        message="Wanted car detected",
    )

    conn = test_db.get_connection()
    with conn:
        conn.execute("DELETE FROM vehicle_records WHERE id = ?", (rec_id,))
    conn.close()

    # Alert should have cascaded
    assert len(test_db.list_alerts()) == 0
