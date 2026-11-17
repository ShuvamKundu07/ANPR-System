"""
anpr/plate_utils.py
───────────────────
Phase 4 — Plate Text Normalization & Format Validation.

Provides:
  • normalize_plate: Cleans, strips, uppercases, and extracts alphanumeric characters.
  • is_valid_indian_plate: Heuristic validation against Indian registration formats.
  • validate_plate: Detailed heuristic inspection with state and format classification.

Important rules:
  • Preserves raw text alongside normalized text.
  • Never silently replaces ambiguous characters (e.g. O/0 or I/1).
  • Validation is explicitly treated as a heuristic, not legal verification.
"""

from __future__ import annotations

import re
from typing import TypedDict

# Two-letter codes for Indian States and Union Territories (Motor Vehicles Act)
INDIAN_STATE_CODES = frozenset(
    {
        "AN",  # Andaman and Nicobar Islands
        "AP",  # Andhra Pradesh
        "AR",  # Arunachal Pradesh
        "AS",  # Assam
        "BR",  # Bihar
        "CG",  # Chhattisgarh
        "CH",  # Chandigarh
        "DD",  # Daman and Diu (historical/merged)
        "DL",  # Delhi
        "DN",  # Dadra and Nagar Haveli
        "GA",  # Goa
        "GJ",  # Gujarat
        "HP",  # Himachal Pradesh
        "HR",  # Haryana
        "JH",  # Jharkhand
        "JK",  # Jammu and Kashmir
        "KA",  # Karnataka
        "KL",  # Kerala
        "LA",  # Ladakh
        "LD",  # Lakshadweep
        "MH",  # Maharashtra
        "ML",  # Meghalaya
        "MN",  # Manipur
        "MP",  # Madhya Pradesh
        "MZ",  # Mizoram
        "NL",  # Nagaland
        "OD",  # Odisha
        "OR",  # Orissa (legacy code)
        "PB",  # Punjab
        "PY",  # Puducherry
        "RJ",  # Rajasthan
        "SK",  # Sikkim
        "TN",  # Tamil Nadu
        "TR",  # Tripura
        "TS",  # Telangana
        "UA",  # Uttarakhand (legacy code)
        "UK",  # Uttarakhand
        "UP",  # Uttar Pradesh
        "WB",  # West Bengal
    }
)

# Standard Indian format: 2-letter state + 1-2 digit RTO + 0-3 letters series + 1-4 digits number
# e.g., DL08AB1234, MH12DE1433, KA011234, DL8CAB1234
_STANDARD_PLATE_REGEX = re.compile(
    r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{1,4})$"
)

# Bharat Series (BH): YY BH #### XX (introduced 2021)
# e.g., 22BH1234AA, 21BH0001B
_BH_SERIES_REGEX = re.compile(r"^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$")


class PlateValidationResult(TypedDict):
    raw: str
    normalized: str
    is_valid: bool
    format: str
    state_code: str | None
    heuristic_note: str


def normalize_plate(raw_text: str | None) -> str:
    """Normalize raw OCR plate text.

    Steps:
      1. Converts to uppercase.
      2. Removes all spaces, hyphens, dots, underscores, and non-alphanumeric separators.
      3. Preserves original characters without silently swapping O/0 or I/1.

    Args:
        raw_text: Raw OCR output string, or None.

    Returns:
        Cleaned alphanumeric uppercase string, or "" if input is empty.
    """
    if not raw_text:
        return ""
    # Strip whitespace, uppercase, keep only letters and numbers
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
    return cleaned


def is_valid_indian_plate(plate_number: str | None) -> bool:
    """Check if normalized plate text matches standard Indian registration patterns.

    Heuristic check only — does not query national registry (VAHAN).

    Args:
        plate_number: Plate string (raw or normalized).

    Returns:
        True if the string matches standard Indian state or BH patterns.
    """
    normalized = normalize_plate(plate_number)
    if not normalized or len(normalized) < 6 or len(normalized) > 12:
        return False

    # Check standard state code format
    match_standard = _STANDARD_PLATE_REGEX.match(normalized)
    if match_standard:
        state_code = match_standard.group(1)
        if state_code in INDIAN_STATE_CODES:
            return True

    # Check BH series format
    match_bh = _BH_SERIES_REGEX.match(normalized)
    if match_bh:
        return True

    return False


def validate_plate(raw_text: str | None) -> PlateValidationResult:
    """Perform detailed heuristic validation of a license plate string.

    Args:
        raw_text: Raw or partial plate text.

    Returns:
        PlateValidationResult dict containing raw text, normalized text,
        validation boolean, format identifier, state code, and explanation.
    """
    raw = "" if raw_text is None else str(raw_text)
    normalized = normalize_plate(raw)

    if not normalized:
        return {
            "raw": raw,
            "normalized": "",
            "is_valid": False,
            "format": "EMPTY",
            "state_code": None,
            "heuristic_note": "Empty or non-alphanumeric input.",
        }

    # Test Bharat Series
    match_bh = _BH_SERIES_REGEX.match(normalized)
    if match_bh:
        return {
            "raw": raw,
            "normalized": normalized,
            "is_valid": True,
            "format": "BHARAT_SERIES",
            "state_code": "BH",
            "heuristic_note": "Matches Bharat Series (BH) vehicle registration format.",
        }

    # Test Standard State Series
    match_std = _STANDARD_PLATE_REGEX.match(normalized)
    if match_std:
        state_code = match_std.group(1)
        if state_code in INDIAN_STATE_CODES:
            return {
                "raw": raw,
                "normalized": normalized,
                "is_valid": True,
                "format": "STANDARD_INDIAN",
                "state_code": state_code,
                "heuristic_note": f"Matches standard Indian format for state/UT: {state_code}.",
            }
        return {
            "raw": raw,
            "normalized": normalized,
            "is_valid": False,
            "format": "UNKNOWN_STATE_CODE",
            "state_code": state_code,
            "heuristic_note": f"Prefix '{state_code}' is not a recognized Indian state or UT code.",
        }

    return {
        "raw": raw,
        "normalized": normalized,
        "is_valid": False,
        "format": "NON_STANDARD",
        "state_code": None,
        "heuristic_note": "Does not conform to standard Indian or BH series registration syntax.",
    }
