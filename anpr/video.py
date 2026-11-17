"""
anpr/video.py
─────────────
Phase 2 — Video and Image Input with OpenCV.

Provides two main entry points:
  • open_source(source)  → VideoCapture-like context manager
  • frame_generator(capture) → iterator of numpy frames

Keeps this module completely independent of YOLO and OCR.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias for a video source: a file path string/Path or a camera index int.
VideoSource = Union[str, Path, int]


class VideoOpenError(RuntimeError):
    """Raised when a video file or camera cannot be opened."""


@contextmanager
def open_source(source: VideoSource) -> Generator[cv2.VideoCapture, None, None]:
    """Open a video file or camera and yield the VideoCapture object.

    Guarantees the capture is released on exit (even on exception).

    Args:
        source: File path (str or Path) for a video file, or an integer camera
                index (e.g. 0 for the default webcam).

    Yields:
        cv2.VideoCapture — an opened capture object.

    Raises:
        VideoOpenError: If the source cannot be opened or is immediately invalid.
        FileNotFoundError: If a file path does not exist on disk.
    """
    # Validate path existence before handing it to OpenCV.
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {p}")
        cap_arg: Union[str, int] = str(p)
    else:
        cap_arg = int(source)

    logger.info("Opening video source: %s", cap_arg)
    cap = cv2.VideoCapture(cap_arg)

    try:
        if not cap.isOpened():
            raise VideoOpenError(
                f"Could not open video source: {cap_arg!r}. "
                "Check the file path, codec support, or camera availability."
            )
        yield cap
    finally:
        cap.release()
        logger.info("VideoCapture released for source: %s", cap_arg)


def frame_generator(
    cap: cv2.VideoCapture,
    max_frames: int = 0,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Iterate frames from an open VideoCapture.

    Args:
        cap: An already-opened cv2.VideoCapture instance.
        max_frames: Stop after this many frames (0 = unlimited / read to EOF).

    Yields:
        (frame_index, frame_bgr) — zero-based index and the BGR NumPy array.

    Notes:
        • Stops cleanly at end-of-file or when the camera feed drops.
        • Logs a warning (not an exception) on a failed read to tolerate
          occasional corrupt frames mid-stream.
    """
    frame_index = 0
    consecutive_failures = 0
    max_consecutive_failures = 5  # tolerate a few transient bad reads

    while True:
        if max_frames and frame_index >= max_frames:
            logger.debug("Reached max_frames limit (%d); stopping.", max_frames)
            break

        ret, frame = cap.read()

        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.info(
                    "Stream ended or %d consecutive read failures; stopping.",
                    consecutive_failures,
                )
                break
            logger.warning(
                "Failed to read frame %d (attempt %d/%d); skipping.",
                frame_index,
                consecutive_failures,
                max_consecutive_failures,
            )
            continue

        consecutive_failures = 0
        yield frame_index, frame
        frame_index += 1


def get_video_properties(cap: cv2.VideoCapture) -> dict:
    """Return basic metadata about an open VideoCapture.

    Args:
        cap: An opened cv2.VideoCapture instance.

    Returns:
        Dictionary with keys: width, height, fps, frame_count.
        frame_count may be -1 if the backend cannot determine it (e.g. webcams).
    """
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

