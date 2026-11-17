"""
anpr/tracking.py
────────────────
Phase 7 — Observation Tracking & Duplicate Suppression.

Provides:
  • DuplicateTracker: Determines whether an observation of (camera_id, plate_number)
    should be recorded or suppressed based on a configurable time window.
  • VehicleJourneyTracker: Aggregates chronological observation history across
    multiple camera feeds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Union

from anpr.alerts import parse_timestamp
from anpr.config import DUPLICATE_WINDOW_SEC
from anpr.database import Database
from anpr.plate_utils import normalize_plate

logger = logging.getLogger(__name__)


class DuplicateTracker:
    """Tracks observations to suppress duplicate database writes across consecutive frames."""

    def __init__(
        self,
        db: Database | None = None,
        window_seconds: int | None = None,
    ) -> None:
        """Initialize tracker with database and suppression window.

        Args:
            db: Database instance.
            window_seconds: Cooldown in seconds. Observations of the same plate on the
                            same camera within this window are considered duplicates.
        """
        self.db = db if db is not None else Database()
        self.window_seconds = (
            window_seconds if window_seconds is not None else DUPLICATE_WINDOW_SEC
        )
        # Fast in-memory cache for high-FPS video streams: (camera_id, plate) -> timestamp
        self._recent_observations: dict[tuple[str, str], datetime] = {}

    def is_duplicate(
        self,
        camera_id: str,
        plate_number: str,
        timestamp: Union[str, datetime, None] = None,
    ) -> bool:
        """Check whether (camera_id, plate_number) is a duplicate within the window.

        Args:
            camera_id: Camera ID (e.g. CAM001).
            plate_number: Normalized or raw plate text.
            timestamp: Observation timestamp.

        Returns:
            True if duplicate (suppress write), False if new observation.
        """
        if self.window_seconds <= 0:
            return False

        norm_plate = normalize_plate(plate_number)
        if not norm_plate:
            return False

        current_dt = parse_timestamp(timestamp)
        cache_key = (camera_id, norm_plate)

        # 1. Check in-memory cache first (instant)
        if cache_key in self._recent_observations:
            last_dt = self._recent_observations[cache_key]
            delta = abs((current_dt - last_dt).total_seconds())
            if delta < self.window_seconds:
                return True

        # 2. Check persistent database
        last_rec = self.db.get_latest_observation(norm_plate, camera_id=camera_id)
        if last_rec:
            last_dt = parse_timestamp(last_rec["detected_at"])
            delta = abs((current_dt - last_dt).total_seconds())
            if delta < self.window_seconds:
                self._recent_observations[cache_key] = last_dt
                return True

        return False

    def record_observation(
        self,
        camera_id: str,
        plate_number: str,
        timestamp: Union[str, datetime, None] = None,
    ) -> None:
        """Update in-memory tracker when a valid observation is saved."""
        norm_plate = normalize_plate(plate_number)
        if norm_plate:
            current_dt = parse_timestamp(timestamp)
            self._recent_observations[(camera_id, norm_plate)] = current_dt

    def clear(self) -> None:
        """Clear the in-memory cache."""
        self._recent_observations.clear()


class VehicleJourneyTracker:
    """Helper to query and format cross-camera vehicle observation journeys."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db if db is not None else Database()

    def get_journey(self, plate_number: str) -> list[dict[str, Any]]:
        """Retrieve chronological observations for a plate across all cameras."""
        return self.db.get_movement_history(plate_number)
