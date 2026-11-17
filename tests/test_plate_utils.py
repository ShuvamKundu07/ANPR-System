"""
tests/test_plate_utils.py
─────────────────────────
Phase 4 — Tests for plate text normalization and validation.

Covers:
  • Normalization of varied inputs (spaces, hyphens, lower-case, punctuation).
  • Preservation of ambiguous characters without silent modification (O vs 0, I vs 1).
  • Validation of standard Indian registration numbers.
  • Validation of Bharat Series (BH) registrations.
  • Rejection of invalid, malformed, or fake state codes.
  • Empty / None inputs handling.
  • Preservation of raw text alongside normalized text.
"""

from __future__ import annotations

import pytest

from anpr.plate_utils import (
    INDIAN_STATE_CODES,
    is_valid_indian_plate,
    normalize_plate,
    validate_plate,
)


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("DL 8C AB 1234", "DL8CAB1234"),
        ("dl 8c ab 1234", "DL8CAB1234"),
        ("MH-12-DE-1433", "MH12DE1433"),
        ("KA.01.E.9999", "KA01E9999"),
        ("  UP 32   AB  0001  ", "UP32AB0001"),
        ("22 BH 1234 AA", "22BH1234AA"),
        ("IND\nHR26DK8337", "INDHR26DK8337"),
        ("", ""),
        (None, ""),
        ("   --- ... ", ""),
    ],
)
def test_normalize_plate(raw, expected):
    """normalize_plate should strip separators and convert to uppercase."""
    assert normalize_plate(raw) == expected


def test_normalize_preserves_ambiguous_characters():
    """Ambiguous characters such as O/0 or I/1 must NOT be silently swapped."""
    # Ensure letter 'O' and digit '0' are not changed into each other
    assert "O" in normalize_plate("MH 01 AB 123O")
    assert "0" in normalize_plate("MH 01 AB 123O")
    # Ensure letter 'I' and digit '1' are preserved
    assert "I" in normalize_plate("DL 1C AI 1234")
    assert "1" in normalize_plate("DL 1C AI 1234")


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plate",
    [
        "DL08AB1234",
        "MH12DE1433",
        "KA01E9999",
        "HR26DK8337",
        "WB02AB1234",
        "TN09B1234",
        "UP32A1",
        "22BH1234AA",
        "21BH0001B",
    ],
)
def test_valid_indian_plates(plate):
    """Valid standard Indian and BH plates must pass heuristic check."""
    assert is_valid_indian_plate(plate) is True


@pytest.mark.parametrize(
    "plate",
    [
        "ZZ08AB1234",  # 'ZZ' is not a valid Indian state code
        "XX9999",      # Missing series and invalid state
        "12345",       # Digits only
        "ABCDEF",      # Letters only
        "",            # Empty
        "TOOLONG123456789",  # Excessive length
    ],
)
def test_invalid_indian_plates(plate):
    """Invalid, out-of-range, or unrecognized state codes must fail validation."""
    assert is_valid_indian_plate(plate) is False


def test_validate_plate_structure():
    """validate_plate returns a comprehensive dictionary with raw and normalized text."""
    res = validate_plate(" dl 8c ab 1234 ")
    assert res["raw"] == " dl 8c ab 1234 "
    assert res["normalized"] == "DL8CAB1234"
    assert res["is_valid"] is True
    assert res["state_code"] == "DL"
    assert res["format"] == "STANDARD_INDIAN"
    assert "Delhi" in res["heuristic_note"] or "DL" in res["heuristic_note"]


def test_validate_plate_bh_series():
    """validate_plate correctly identifies Bharat (BH) series."""
    res = validate_plate("22 BH 1234 AA")
    assert res["normalized"] == "22BH1234AA"
    assert res["is_valid"] is True
    assert res["format"] == "BHARAT_SERIES"
    assert res["state_code"] == "BH"


def test_validate_plate_empty():
    """validate_plate handles empty input safely."""
    res = validate_plate(None)
    assert res["normalized"] == ""
    assert res["is_valid"] is False
    assert res["format"] == "EMPTY"
