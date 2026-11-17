"""
anpr/database.py
────────────────
Phase 5 — SQLite Database Operations.

Manages tables:
  • vehicle_records: Observations from video / camera feeds.
  • hotlist: Flagged license plates with status and notes.
  • alerts: Generated alert notifications linked to vehicle observations.

Guarantees:
  • Automatic schema migration/creation on first start.
  • 100% parameterized queries (no SQL injection).
  • Isolation for testing with temporary or in-memory databases.
  • Safe relative image path storage.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from anpr.config import DB_PATH, PROJECT_ROOT
from anpr.plate_utils import normalize_plate

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Database:
    """SQLite database manager for ANPR POC observations, hotlist, and alerts."""

    def __init__(self, db_path: Union[str, Path, None] = None) -> None:
        """Initialize database connection manager and ensure tables exist.

        Args:
            db_path: Path to SQLite file, or ':memory:'. Defaults to config.DB_PATH.
        """
        if db_path == ":memory:":
            self.db_path = ":memory:"
            # In-memory database requires a persistent connection
            self._mem_conn: sqlite3.Connection | None = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            self.db_path = Path(db_path) if db_path else DB_PATH
            self._mem_conn = None
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing ANPR database at: %s", self.db_path)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Get a SQLite connection with row_factory set to sqlite3.Row."""
        if self._mem_conn is not None:
            return self._mem_conn

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Create tables and indexes if they do not already exist."""
        conn = self.get_connection()
        try:
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS vehicle_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_text_raw TEXT NOT NULL,
                        plate_number TEXT NOT NULL,
                        ocr_confidence REAL NOT NULL,
                        detected_at TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        detection_confidence REAL NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_records_plate 
                        ON vehicle_records(plate_number);
                    CREATE INDEX IF NOT EXISTS idx_records_cam_time 
                        ON vehicle_records(camera_id, detected_at);
                    CREATE INDEX IF NOT EXISTS idx_records_time 
                        ON vehicle_records(detected_at);

                    CREATE TABLE IF NOT EXISTS hotlist (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT UNIQUE NOT NULL,
                        status TEXT NOT NULL,
                        description TEXT,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_hotlist_plate 
                        ON hotlist(plate_number);

                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        vehicle_record_id INTEGER NOT NULL,
                        plate_number TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        acknowledged INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY (vehicle_record_id) 
                            REFERENCES vehicle_records(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_alerts_plate 
                        ON alerts(plate_number);
                    CREATE INDEX IF NOT EXISTS idx_alerts_ack 
                        ON alerts(acknowledged);
                    """
                )
            logger.info("Database schema initialized successfully.")
        finally:
            if self._mem_conn is None:
                conn.close()

    # ── Path Helper ───────────────────────────────────────────────────────────

    @staticmethod
    def to_relative_image_path(image_path: Union[str, Path]) -> str:
        """Convert an absolute image path to a project-relative path where practical."""
        p = Path(image_path)
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    @staticmethod
    def to_absolute_image_path(rel_path: str) -> Path:
        """Convert a stored image path back to an absolute Path."""
        p = Path(rel_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    # ── Vehicle Records Operations ────────────────────────────────────────────

    def insert_observation(
        self,
        plate_text_raw: str,
        plate_number: str,
        ocr_confidence: float,
        camera_id: str,
        image_path: Union[str, Path],
        detection_confidence: float,
        detected_at: Union[datetime, str, None] = None,
    ) -> int:
        """Insert a vehicle observation record.

        Args:
            plate_text_raw: Exact OCR output text before cleaning.
            plate_number: Normalized registration number.
            ocr_confidence: OCR confidence score (0.0 - 1.0).
            camera_id: Identifier of the source camera (e.g. CAM001).
            image_path: Path to the saved plate crop.
            detection_confidence: YOLO plate detector confidence score.
            detected_at: Timestamp (defaults to current UTC ISO string).

        Returns:
            The integer primary key (id) of the inserted record.
        """
        if isinstance(detected_at, datetime):
            timestamp_str = detected_at.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(detected_at, str):
            timestamp_str = detected_at
        else:
            timestamp_str = utc_now_iso()

        rel_img = self.to_relative_image_path(image_path)
        norm_plate = normalize_plate(plate_number)

        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO vehicle_records (
                        plate_text_raw,
                        plate_number,
                        ocr_confidence,
                        detected_at,
                        camera_id,
                        image_path,
                        detection_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plate_text_raw,
                        norm_plate,
                        float(ocr_confidence),
                        timestamp_str,
                        str(camera_id),
                        rel_img,
                        float(detection_confidence),
                    ),
                )
                record_id = cursor.lastrowid
                assert record_id is not None
                return record_id
        finally:
            if self._mem_conn is None:
                conn.close()

    def get_recent_observations(
        self, limit: int = 50, camera_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve recent vehicle observations ordered by time descending."""
        conn = self.get_connection()
        try:
            if camera_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM vehicle_records 
                    WHERE camera_id = ? 
                    ORDER BY detected_at DESC, id DESC 
                    LIMIT ?
                    """,
                    (camera_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM vehicle_records 
                    ORDER BY detected_at DESC, id DESC 
                    LIMIT ?
                    """,
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            if self._mem_conn is None:
                conn.close()

    def search_observations(
        self,
        plate_number: str | None = None,
        camera_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search observations by plate number, camera ID, and date/time range."""
        query = "SELECT * FROM vehicle_records WHERE 1=1"
        params: list[Any] = []

        if plate_number:
            query += " AND plate_number LIKE ?"
            params.append(f"%{normalize_plate(plate_number)}%")

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)

        if start_time:
            query += " AND detected_at >= ?"
            params.append(start_time)

        if end_time:
            query += " AND detected_at <= ?"
            params.append(end_time)

        query += " ORDER BY detected_at DESC, id DESC LIMIT ?"
        params.append(limit)

        conn = self.get_connection()
        try:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            if self._mem_conn is None:
                conn.close()

    def get_movement_history(self, plate_number: str) -> list[dict[str, Any]]:
        """Retrieve full chronological observation history for a given plate."""
        norm_plate = normalize_plate(plate_number)
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM vehicle_records 
                WHERE plate_number = ? 
                ORDER BY detected_at ASC, id ASC
                """,
                (norm_plate,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            if self._mem_conn is None:
                conn.close()

    def get_latest_observation(
        self, plate_number: str, camera_id: str | None = None
    ) -> dict[str, Any] | None:
        """Get the most recent observation for a plate (useful for duplicate suppression)."""
        norm_plate = normalize_plate(plate_number)
        conn = self.get_connection()
        try:
            if camera_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM vehicle_records 
                    WHERE plate_number = ? AND camera_id = ?
                    ORDER BY detected_at DESC, id DESC LIMIT 1
                    """,
                    (norm_plate, camera_id),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM vehicle_records 
                    WHERE plate_number = ? 
                    ORDER BY detected_at DESC, id DESC LIMIT 1
                    """,
                    (norm_plate,),
                )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            if self._mem_conn is None:
                conn.close()

    # ── Hotlist Operations ────────────────────────────────────────────────────

    def add_hotlist_entry(
        self,
        plate_number: str,
        status: str,
        description: str | None = None,
        created_at: str | None = None,
    ) -> int:
        """Add or update a hotlist entry with normalized plate number."""
        norm_plate = normalize_plate(plate_number)
        if not norm_plate:
            raise ValueError("Cannot add an empty plate number to the hotlist.")

        ts = created_at or utc_now_iso()
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO hotlist (plate_number, status, description, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(plate_number) DO UPDATE SET
                        status = excluded.status,
                        description = excluded.description,
                        created_at = excluded.created_at
                    """,
                    (norm_plate, status.upper().strip(), description, ts),
                )
                entry_id = cursor.lastrowid
                assert entry_id is not None
                return entry_id
        finally:
            if self._mem_conn is None:
                conn.close()

    def list_hotlist_entries(self) -> list[dict[str, Any]]:
        """List all hotlist entries sorted by plate number."""
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM hotlist ORDER BY plate_number ASC"
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            if self._mem_conn is None:
                conn.close()

    def get_hotlist_entry(self, plate_number: str) -> dict[str, Any] | None:
        """Look up a specific hotlist entry by plate number."""
        norm_plate = normalize_plate(plate_number)
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM hotlist WHERE plate_number = ?",
                (norm_plate,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            if self._mem_conn is None:
                conn.close()

    def remove_hotlist_entry(self, plate_number: str) -> bool:
        """Delete an entry from the hotlist. Returns True if deleted, False otherwise."""
        norm_plate = normalize_plate(plate_number)
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM hotlist WHERE plate_number = ?",
                    (norm_plate,),
                )
                return cursor.rowcount > 0
        finally:
            if self._mem_conn is None:
                conn.close()

    # ── Alerts Operations ─────────────────────────────────────────────────────

    def create_alert(
        self,
        vehicle_record_id: int,
        plate_number: str,
        alert_type: str,
        message: str,
        created_at: str | None = None,
        acknowledged: bool = False,
    ) -> int:
        """Create a new alert linked to a vehicle record."""
        norm_plate = normalize_plate(plate_number)
        ts = created_at or utc_now_iso()
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO alerts (
                        vehicle_record_id,
                        plate_number,
                        alert_type,
                        message,
                        created_at,
                        acknowledged
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(vehicle_record_id),
                        norm_plate,
                        alert_type.strip(),
                        message.strip(),
                        ts,
                        1 if acknowledged else 0,
                    ),
                )
                alert_id = cursor.lastrowid
                assert alert_id is not None
                return alert_id
        finally:
            if self._mem_conn is None:
                conn.close()

    def list_alerts(
        self, acknowledged: bool | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List alerts joined with vehicle observation metadata."""
        query = """
            SELECT 
                a.id,
                a.vehicle_record_id,
                a.plate_number,
                a.alert_type,
                a.message,
                a.created_at,
                a.acknowledged,
                v.camera_id,
                v.image_path,
                v.detected_at,
                v.ocr_confidence
            FROM alerts a
            JOIN vehicle_records v ON a.vehicle_record_id = v.id
        """
        params: list[Any] = []

        if acknowledged is not None:
            query += " WHERE a.acknowledged = ?"
            params.append(1 if acknowledged else 0)

        query += " ORDER BY a.created_at DESC, a.id DESC LIMIT ?"
        params.append(limit)

        conn = self.get_connection()
        try:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            if self._mem_conn is None:
                conn.close()

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged."""
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
                    (int(alert_id),),
                )
                return cursor.rowcount > 0
        finally:
            if self._mem_conn is None:
                conn.close()

    def get_alert(self, alert_id: int) -> dict[str, Any] | None:
        """Fetch a single alert by ID."""
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT 
                    a.*, 
                    v.camera_id, 
                    v.image_path, 
                    v.detected_at
                FROM alerts a
                JOIN vehicle_records v ON a.vehicle_record_id = v.id
                WHERE a.id = ?
                """,
                (int(alert_id),),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            if self._mem_conn is None:
                conn.close()
