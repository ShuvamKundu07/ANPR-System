"""
anpr/detector.py
────────────────
Phase 3 — License Plate Detection with YOLO.

Provides:
  • Detection dataclass: coordinates, confidence, class metadata
  • LicensePlateDetector: loads weights, runs inference, draws annotations,
    crops plates, and saves crops.

Keeps model loading separate from frame processing and guarantees
configurable confidence thresholds and safe bounding box clamping.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np

# Set Ultralytics config dir to /tmp if default is not writable
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from ultralytics import YOLO  # noqa: E402

from anpr.config import CROP_DIR, DETECTOR_CONFIDENCE, MODEL_PATH  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MODEL_URL: str = (
    "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8"
    "/raw/main/license_plate_detector.pt"
)


def ensure_model_weights(dest_path: Union[str, Path, None] = None) -> Path:
    """Ensure YOLO detection model weights exist, downloading if missing.

    Args:
        dest_path: Target path for .pt file. Defaults to config.MODEL_PATH.

    Returns:
        Path to the available weights file.
    """
    target = Path(dest_path) if dest_path else MODEL_PATH
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Model weights missing at %s; downloading from approved repository...", target)
    import urllib.request

    urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(target))
    logger.info("Successfully downloaded YOLO weights to %s (size: %d bytes)", target, target.stat().st_size)
    return target


@dataclass(frozen=True)
class Detection:
    """Represents a detected license plate bounding box."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0
    class_name: str = "license_plate"

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0 and 0.0 <= self.confidence <= 1.0


class LicensePlateDetector:
    """Wrapper around YOLO license-plate detection model."""

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        confidence_threshold: float | None = None,
    ) -> None:
        """Initialize and load the YOLO detection model once.

        Args:
            model_path: Path to the .pt model weights file. Defaults to config.MODEL_PATH.
            confidence_threshold: Minimum confidence score. Defaults to config.DETECTOR_CONFIDENCE.

        Raises:
            FileNotFoundError: If the specified model_path does not exist.
        """
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"License plate detector model not found at: {self.model_path}. "
                "Ensure weights are downloaded or configure MODEL_PATH in .env."
            )

        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else DETECTOR_CONFIDENCE
        )

        logger.info(
            "Loading YOLO license plate detector from %s (threshold=%.2f)",
            self.model_path,
            self.confidence_threshold,
        )
        self.model = YOLO(str(self.model_path))
        logger.info("YOLO model loaded successfully. Classes: %s", self.model.names)

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[Detection]:
        """Detect license plates in a single frame.

        Args:
            frame: BGR image array (NumPy array).
            confidence_threshold: Optional override for detection confidence threshold.

        Returns:
            List of valid Detection objects (empty if none found or frame invalid).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Empty or invalid frame passed to detect(); returning [].")
            return []

        conf = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )
        h, w = frame.shape[:2]

        results = self.model(frame, conf=conf, verbose=False)
        detections: list[Detection] = []

        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            score = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else 0
            cls_name = self.model.names.get(cls_id, "license_plate")

            # Extract coordinates and clamp to frame boundaries
            raw_xyxy = box.xyxy[0].tolist()
            x1 = max(0, min(w, int(round(raw_xyxy[0]))))
            y1 = max(0, min(h, int(round(raw_xyxy[1]))))
            x2 = max(0, min(w, int(round(raw_xyxy[2]))))
            y2 = max(0, min(h, int(round(raw_xyxy[3]))))

            det = Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=score,
                class_id=cls_id,
                class_name=cls_name,
            )

            if det.is_valid():
                detections.append(det)
            else:
                logger.debug("Rejected invalid/zero-area detection: %s", det)

        return detections

    @staticmethod
    def crop_plate(frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        """Crop the bounding box of a detected license plate from the frame.

        Clamps coordinates to frame bounds and returns None if crop is empty.

        Args:
            frame: Source image (BGR NumPy array).
            detection: Detection object containing coordinates.

        Returns:
            Cropped image as a NumPy array, or None if crop is invalid.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        x1 = max(0, min(w, detection.x1))
        y1 = max(0, min(h, detection.y1))
        x2 = max(0, min(w, detection.x2))
        y2 = max(0, min(h, detection.y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop

    @staticmethod
    def save_crop(
        crop: np.ndarray,
        output_dir: Union[str, Path, None] = None,
        filename: str | None = None,
    ) -> Path:
        """Save a cropped plate image to disk.

        Args:
            crop: Cropped plate image as NumPy array.
            output_dir: Target directory. Defaults to config.CROP_DIR.
            filename: Target file name. If None, generates an ISO timestamp + UUID name.

        Returns:
            Path to the saved image file.

        Raises:
            ValueError: If crop is empty or invalid.
            IOError: If OpenCV fails to write the image.
        """
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            raise ValueError("Cannot save an empty or invalid crop.")

        target_dir = Path(output_dir) if output_dir else CROP_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:8]
            filename = f"plate_{ts}_{uid}.jpg"

        file_path = target_dir / filename
        success = cv2.imwrite(str(file_path), crop)
        if not success:
            raise OSError(f"cv2.imwrite failed to save crop to: {file_path}")

        logger.debug("Saved plate crop to %s", file_path)
        return file_path

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: Sequence[Detection],
        copy: bool = True,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and confidence labels on a frame.

        Args:
            frame: Input image array.
            detections: List/sequence of Detection objects.
            copy: If True, operates on a copy; if False, draws in-place.
            color: BGR color for bounding box (default green: (0, 255, 0)).
            thickness: Line thickness for bounding box.

        Returns:
            Annotated frame array.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a valid NumPy array.")

        out = frame.copy() if copy else frame

        for det in detections:
            cv2.rectangle(
                out,
                (det.x1, det.y1),
                (det.x2, det.y2),
                color,
                thickness,
            )
            label = f"{det.class_name} {det.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            text_thick = 1

            # Compute text size for background banner
            (tw, th), baseline = cv2.getTextSize(label, font, scale, text_thick)
            ty1 = max(0, det.y1 - th - baseline - 4)
            ty2 = det.y1

            # Background rectangle for text readability
            cv2.rectangle(out, (det.x1, ty1), (det.x1 + tw + 4, ty2), color, -1)
            cv2.putText(
                out,
                label,
                (det.x1 + 2, ty2 - baseline),
                font,
                scale,
                (0, 0, 0),  # black text on green background
                text_thick,
                cv2.LINE_AA,
            )

        return out
