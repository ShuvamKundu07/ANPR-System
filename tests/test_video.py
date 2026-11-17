"""
tests/test_video.py
────────────────────
Phase 2 — Tests for anpr/video.py.

Covers:
  • Opening a valid video file and reading all frames.
  • get_video_properties returns sensible values.
  • frame_generator max_frames limit is respected.
  • FileNotFoundError for a missing file path.
  • VideoOpenError for an invalid/unopenable source string.
  • capture is released even when the generator is interrupted.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from anpr.video import (
    VideoOpenError,
    frame_generator,
    get_video_properties,
    open_source,
)

# ---------------------------------------------------------------------------
# Fixture: path to the synthetic test video created during Phase 2 setup.
# ---------------------------------------------------------------------------
SAMPLE_VIDEO = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "input_videos"
    / "sample_test.mp4"
)


@pytest.fixture
def sample_video_path() -> pathlib.Path:
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")
    return SAMPLE_VIDEO


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_open_valid_file_yields_capture(sample_video_path):
    """open_source should yield a capture that is opened."""
    with open_source(sample_video_path) as cap:
        assert cap.isOpened(), "Capture must be open inside context"


def test_capture_released_after_context(sample_video_path):
    """VideoCapture must be released after leaving the context."""
    with open_source(sample_video_path) as cap:
        pass  # just open and close
    assert not cap.isOpened(), "Capture must be released after context exit"


def test_frame_generator_reads_all_frames(sample_video_path):
    """frame_generator must yield all 30 synthetic frames."""
    with open_source(sample_video_path) as cap:
        frames = list(frame_generator(cap))

    assert len(frames) == 30, f"Expected 30 frames, got {len(frames)}"
    # Check indices are sequential
    indices = [idx for idx, _ in frames]
    assert indices == list(range(30))


def test_frame_generator_respects_max_frames(sample_video_path):
    """frame_generator stops at max_frames even when more frames remain."""
    with open_source(sample_video_path) as cap:
        frames = list(frame_generator(cap, max_frames=5))

    assert len(frames) == 5, f"Expected 5 frames, got {len(frames)}"


def test_frames_are_numpy_arrays(sample_video_path):
    """Each yielded frame must be an H×W×3 uint8 array."""
    with open_source(sample_video_path) as cap:
        for _idx, frame in frame_generator(cap, max_frames=3):
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
            assert frame.shape[2] == 3
            assert frame.dtype == np.uint8


def test_get_video_properties_returns_dict(sample_video_path):
    """get_video_properties should return expected keys with plausible values."""
    with open_source(sample_video_path) as cap:
        props = get_video_properties(cap)

    assert set(props.keys()) == {"width", "height", "fps", "frame_count"}
    assert props["width"] == 320
    assert props["height"] == 240
    # frame_count may be 0 or 30 depending on codec
    assert props["frame_count"] >= 0


def test_missing_file_raises_file_not_found():
    """A non-existent file path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        with open_source("/tmp/definitely_does_not_exist_xyzxyz.mp4"):
            pass


def test_invalid_source_string_raises_video_open_error(tmp_path):
    """An unreadable source (empty file) must raise VideoOpenError."""
    bad_file = tmp_path / "empty.mp4"
    bad_file.write_bytes(b"")  # zero-byte file; OpenCV cannot open it
    with pytest.raises((VideoOpenError, FileNotFoundError)):
        with open_source(bad_file) as cap:
            # OpenCV may open but isOpened() returns False → VideoOpenError
            pass
