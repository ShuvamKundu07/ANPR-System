"""
anpr/hotlist.py
───────────────
Phase 6 — Hotlist Matching.

Provides:
  • HotlistMatcher: queries hotlist entries in SQLite for incoming plates.
  • Strict normalized string matching (fuzzy matching disabled by default to
    avoid false positive alerts).
"""

from __future__ import annotations

import logging
from typing import Any

from anpr.database import Database
from anpr.plate_utils import normalize_plate

logger = logging.getLogger(__name__)


class HotlistMatcher:
    """Matches recognized vehicle license plates against the database hotlist."""

    def __init__(self, db: Database | None = None) -> None:
        """Initialize matcher with database instance.

        Args:
            db: Database instance. If None, instantiates a default Database.
        """
        self.db = db if db is not None else Database()

    def check(self, plate_number: str | None) -> dict[str, Any] | None:
        """Check if a license plate matches any hotlist entry.

        Normalizes the incoming plate using the same function used for hotlist
        storage, and performs an exact string comparison.

        Args:
            plate_number: Raw or normalized plate string.

        Returns:
            Dictionary with hotlist entry details if matched, or None.
        """
        norm_plate = normalize_plate(plate_number)
        if not norm_plate:
            return None

        entry = self.db.get_hotlist_entry(norm_plate)
        if entry:
            logger.warning(
                "HOTLIST MATCH detected for plate: %s (Status: %s)",
                norm_plate,
                entry.get("status"),
            )
            return entry

        logger.debug("Plate %s is not in hotlist.", norm_plate)
        return None

    def is_hotlisted(self, plate_number: str | None) -> bool:
        """Return True if plate is in hotlist, False otherwise."""
        return self.check(plate_number) is not None

    def add_to_hotlist(
        self,
        plate_number: str,
        status: str,
        description: str | None = None,
    ) -> int:
        """Convenience method to add or update an entry in the hotlist."""
        return self.db.add_hotlist_entry(
            plate_number=plate_number,
            status=status,
            description=description,
        )

    def remove_from_hotlist(self, plate_number: str) -> bool:
        """Convenience method to remove an entry from the hotlist."""
        return self.db.remove_hotlist_entry(plate_number)
