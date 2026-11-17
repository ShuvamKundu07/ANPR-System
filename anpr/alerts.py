"""
anpr/alerts.py
──────────────
Phase 6 — Alerts Management and Duplicate Suppression.

Provides:
  • AlertManager: creates, logs, suppresses duplicates, and manages alerts.
  • Prevents alert floods across consecutive video frames showing the same vehicle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Union

from anpr.config import DUPLICATE_WINDOW_SEC
from anpr.database import Database, utc_now_iso
from anpr.plate_utils import normalize_plate

logger = logging.getLogger(__name__)


def parse_timestamp(ts: Union[str, datetime, None]) -> datetime:
    """Parse a timestamp string or datetime into a UTC datetime object."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if not ts:
        return datetime.now(timezone.utc)

    # Replace space with T for ISO format parsing if needed
    cleaned = ts.strip().replace(" ", "T")
    try:
        dt = datetime.fromisoformat(cleaned)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        # Fallback to standard SQL format
        try:
            dt = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)


class AlertManager:
    """Manages alert creation and duplicate alert suppression."""

    def __init__(
        self,
        db: Database | None = None,
        duplicate_window_sec: int | None = None,
    ) -> None:
        """Initialize AlertManager with database and suppression window.

        Args:
            db: Database instance. If None, instantiates a default Database.
            duplicate_window_sec: Window in seconds to suppress duplicate alerts for
                                  the same plate and camera. Defaults to config.DUPLICATE_WINDOW_SEC.
        """
        self.db = db if db is not None else Database()
        self.duplicate_window_sec = (
            duplicate_window_sec
            if duplicate_window_sec is not None
            else DUPLICATE_WINDOW_SEC
        )

    def is_duplicate_alert(
        self,
        plate_number: str,
        camera_id: str,
        current_time: Union[str, datetime, None] = None,
    ) -> bool:
        """Check if an alert for this plate and camera was created within duplicate_window_sec.

        Args:
            plate_number: Plate string (will be normalized).
            camera_id: Camera identifier.
            current_time: Time of current observation.

        Returns:
            True if an alert should be suppressed (duplicate), False otherwise.
        """
        if self.duplicate_window_sec <= 0:
            return False

        norm_plate = normalize_plate(plate_number)
        now_dt = parse_timestamp(current_time)

        conn = self.db.get_connection()
        try:
            # Query the latest alert timestamp for this plate and camera
            cursor = conn.execute(
                """
                SELECT a.created_at 
                FROM alerts a
                JOIN vehicle_records v ON a.vehicle_record_id = v.id
                WHERE a.plate_number = ? AND v.camera_id = ?
                ORDER BY a.created_at DESC, a.id DESC LIMIT 1
                """,
                (norm_plate, camera_id),
            )
            row = cursor.fetchone()
            if not row:
                return False

            last_alert_dt = parse_timestamp(row["created_at"])
            delta_seconds = abs((now_dt - last_alert_dt).total_seconds())

            if delta_seconds < self.duplicate_window_sec:
                logger.info(
                    "Duplicate alert suppressed for plate %s on camera %s "
                    "(last alert was %.1fs ago, window is %ds).",
                    norm_plate,
                    camera_id,
                    delta_seconds,
                    self.duplicate_window_sec,
                )
                return True

            return False
        finally:
            if self.db._mem_conn is None:
                conn.close()

    def create_hotlist_alert(
        self,
        vehicle_record_id: int,
        plate_number: str,
        camera_id: str,
        hotlist_entry: dict[str, Any],
        image_path: str,
        detected_at: Union[str, datetime, None] = None,
    ) -> dict[str, Any] | None:
        """Create an alert for a hotlist match if not suppressed by duplicate window.

        Args:
            vehicle_record_id: Primary key of observation in vehicle_records.
            plate_number: Vehicle license plate.
            camera_id: Camera ID where vehicle was observed.
            hotlist_entry: Dictionary containing hotlist entry metadata (status, description).
            image_path: Path to the saved plate crop.
            detected_at: Timestamp of observation.

        Returns:
            Created alert dictionary, or None if suppressed as a duplicate.
        """
        norm_plate = normalize_plate(plate_number)
        ts = detected_at if isinstance(detected_at, str) else utc_now_iso()

        # Check duplicate suppression window
        if self.is_duplicate_alert(norm_plate, camera_id, ts):
            return None

        status = hotlist_entry.get("status", "FLAGGED").upper()
        description = hotlist_entry.get("description") or "No description provided."
        message = (
            f"HOTLIST MATCH [{status}]: Plate {norm_plate} detected by {camera_id}. "
            f"Details: {description}"
        )

        alert_id = self.db.create_alert(
            vehicle_record_id=vehicle_record_id,
            plate_number=norm_plate,
            alert_type="HOTLIST_MATCH",
            message=message,
            created_at=ts,
            acknowledged=False,
        )

        logger.warning(
            "ALERT GENERATED (ID: %d): Plate %s [Status: %s] at %s",
            alert_id,
            norm_plate,
            status,
            camera_id,
        )

        return {
            "id": alert_id,
            "vehicle_record_id": vehicle_record_id,
            "plate_number": norm_plate,
            "alert_type": "HOTLIST_MATCH",
            "status": status,
            "message": message,
            "camera_id": camera_id,
            "image_path": str(image_path),
            "created_at": ts,
            "acknowledged": 0,
        }

    def list_alerts(
        self, acknowledged: bool | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List recent alerts joined with vehicle observation metadata."""
        return self.db.list_alerts(acknowledged=acknowledged, limit=limit)

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged by user."""
        return self.db.acknowledge_alert(alert_id)

    def get_alert(self, alert_id: int) -> dict[str, Any] | None:
        """Retrieve a specific alert by ID."""
        return self.db.get_alert(alert_id)
