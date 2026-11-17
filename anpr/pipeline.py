"""
anpr/pipeline.py
────────────────
Phase 7 — End-to-End ANPR Pipeline.

Orchestrates:
  1. Video frame input (via anpr.video)
  2. License plate detection (via anpr.detector)
  3. Cropping and OCR extraction (via anpr.ocr)
  4. Text normalization and validation (via anpr.plate_utils)
  5. Duplicate observation suppression (via anpr.tracking)
  6. Persistent storage in SQLite (via anpr.database)
  7. Hotlist matching and alert dispatch (via anpr.hotlist & anpr.alerts)
  8. Annotated frame rendering with color-coded status banners
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generator, Sequence, Union

import cv2
import numpy as np

from anpr.alerts import AlertManager, utc_now_iso
from anpr.config import (
    CROP_DIR,
    DEFAULT_CAMERA_ID,
    DETECTOR_CONFIDENCE,
    DUPLICATE_WINDOW_SEC,
    OCR_CONFIDENCE_THRESHOLD,
)
from anpr.database import Database
from anpr.detector import Detection, LicensePlateDetector
from anpr.hotlist import HotlistMatcher
from anpr.ocr import OCRResult, PlateOCR
from anpr.plate_utils import PlateValidationResult, normalize_plate, validate_plate
from anpr.tracking import DuplicateTracker
from anpr.video import VideoSource, frame_generator, open_source

logger = logging.getLogger(__name__)


@dataclass
class PipelineObservation:
    """Represents a single recognized plate observation within a frame."""

    detection: Detection
    ocr_result: OCRResult
    plate_text_raw: str
    plate_number: str
    is_valid_plate: bool
    validation: PlateValidationResult
    is_duplicate: bool = False
    is_hotlist_match: bool = False
    hotlist_entry: dict[str, Any] | None = None
    alert: dict[str, Any] | None = None
    db_record_id: int | None = None
    image_path: Path | None = None


@dataclass
class PipelineResult:
    """Represents the complete pipeline analysis for one processed frame."""

    frame_index: int
    annotated_frame: np.ndarray
    detections: list[Detection]
    observations: list[PipelineObservation] = field(default_factory=list)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0

    @property
    def has_valid_observations(self) -> bool:
        return any(obs.is_valid_plate for obs in self.observations)

    @property
    def has_hotlist_alerts(self) -> bool:
        return any(obs.alert is not None for obs in self.observations)


class ANPRPipeline:
    """End-to-End Automatic License Plate Recognition Pipeline."""

    def __init__(
        self,
        camera_id: str = "CAM001",
        db: Database | None = None,
        detector: LicensePlateDetector | None = None,
        ocr: PlateOCR | None = None,
        crop_dir: Union[str, Path, None] = None,
        detector_conf: float = DETECTOR_CONFIDENCE,
        ocr_conf_thresh: float = OCR_CONFIDENCE_THRESHOLD,
        duplicate_window_sec: int = DUPLICATE_WINDOW_SEC,
    ) -> None:
        """Initialize the pipeline with all vision, OCR, and storage engines.

        Models and database connections are initialized once and reused.
        """
        self.camera_id = str(camera_id)
        self.db = db if db is not None else Database()
        self.detector = (
            detector
            if detector is not None
            else LicensePlateDetector(confidence_threshold=detector_conf)
        )
        self.ocr = (
            ocr
            if ocr is not None
            else PlateOCR(confidence_threshold=ocr_conf_thresh)
        )
        self.matcher = HotlistMatcher(db=self.db)
        self.alert_manager = AlertManager(
            db=self.db, duplicate_window_sec=duplicate_window_sec
        )
        self.duplicate_tracker = DuplicateTracker(
            db=self.db, window_seconds=duplicate_window_sec
        )
        self.crop_dir = Path(crop_dir) if crop_dir else CROP_DIR
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_conf_thresh = ocr_conf_thresh

        logger.info(
            "ANPRPipeline initialized (camera_id=%s, duplicate_window=%ds)",
            self.camera_id,
            duplicate_window_sec,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: Union[str, datetime, None] = None,
        camera_id: str | None = None,
    ) -> PipelineResult:
        """Process a single video or camera frame through the full ANPR pipeline.

        Args:
            frame: BGR NumPy array.
            frame_index: Sequential frame number.
            timestamp: Time of capture (defaults to current UTC ISO string).
            camera_id: Override camera identifier if processing multi-camera feeds.

        Returns:
            PipelineResult containing annotated frame, detections, and observations.
        """
        active_camera_id = camera_id or self.camera_id
        ts_str = timestamp if isinstance(timestamp, str) else utc_now_iso()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return PipelineResult(
                frame_index=frame_index,
                annotated_frame=np.zeros((100, 100, 3), dtype=np.uint8),
                detections=[],
                observations=[],
            )

        annotated = frame.copy()

        # Step 1: Detect license plates
        detections = self.detector.detect(frame)
        observations: list[PipelineObservation] = []

        if not detections:
            return PipelineResult(
                frame_index=frame_index,
                annotated_frame=annotated,
                detections=[],
                observations=[],
            )

        # Step 2: Process each detected plate
        for det in detections:
            crop = self.detector.crop_plate(frame, det)
            if crop is None:
                continue

            # Step 3: Run OCR on cropped plate
            ocr_res = self.ocr.read_plate(crop)
            raw_text = ocr_res.raw_text
            normalized_plate = normalize_plate(raw_text)
            validation = validate_plate(raw_text)

            is_valid_reading = (
                not ocr_res.is_empty
                and bool(normalized_plate)
                and ocr_res.confidence >= self.ocr_conf_thresh
            )

            is_duplicate = False
            is_hotlist = False
            hotlist_entry = None
            alert = None
            db_record_id = None
            saved_crop_path = None

            if is_valid_reading:
                # Step 4: Check hotlist
                hotlist_entry = self.matcher.check(normalized_plate)
                is_hotlist = hotlist_entry is not None

                # Step 5: Duplicate observation check
                is_duplicate = self.duplicate_tracker.is_duplicate(
                    camera_id=active_camera_id,
                    plate_number=normalized_plate,
                    timestamp=ts_str,
                )

                if not is_duplicate:
                    # Save crop to disk
                    saved_crop_path = self.detector.save_crop(
                        crop, output_dir=self.crop_dir
                    )

                    # Insert observation into SQLite
                    db_record_id = self.db.insert_observation(
                        plate_text_raw=raw_text,
                        plate_number=normalized_plate,
                        ocr_confidence=ocr_res.confidence,
                        camera_id=active_camera_id,
                        image_path=saved_crop_path,
                        detection_confidence=det.confidence,
                        detected_at=ts_str,
                    )

                    # Update duplicate tracker
                    self.duplicate_tracker.record_observation(
                        camera_id=active_camera_id,
                        plate_number=normalized_plate,
                        timestamp=ts_str,
                    )

                    # Step 6: Create alert if hotlist matched
                    if is_hotlist and hotlist_entry is not None and db_record_id:
                        alert = self.alert_manager.create_hotlist_alert(
                            vehicle_record_id=db_record_id,
                            plate_number=normalized_plate,
                            camera_id=active_camera_id,
                            hotlist_entry=hotlist_entry,
                            image_path=str(saved_crop_path),
                            detected_at=ts_str,
                        )

            obs = PipelineObservation(
                detection=det,
                ocr_result=ocr_res,
                plate_text_raw=raw_text,
                plate_number=normalized_plate,
                is_valid_plate=is_valid_reading,
                validation=validation,
                is_duplicate=is_duplicate,
                is_hotlist_match=is_hotlist,
                hotlist_entry=hotlist_entry,
                alert=alert,
                db_record_id=db_record_id,
                image_path=saved_crop_path,
            )
            observations.append(obs)

            # Step 7: Render bounding box and label on annotated frame
            self._draw_observation_annotation(annotated, obs)

        return PipelineResult(
            frame_index=frame_index,
            annotated_frame=annotated,
            detections=detections,
            observations=observations,
        )

    def process_video(
        self,
        source: VideoSource,
        max_frames: int = 0,
        frame_callback: Callable[[PipelineResult], None] | None = None,
    ) -> Generator[PipelineResult, None, None]:
        """Process an entire video stream or camera feed frame by frame.

        Args:
            source: Path to video file or integer webcam index.
            max_frames: Max frames to process (0 = all).
            frame_callback: Optional hook called with PipelineResult on each frame.

        Yields:
            PipelineResult for each processed frame.
        """
        with open_source(source) as cap:
            for idx, frame in frame_generator(cap, max_frames=max_frames):
                res = self.process_frame(frame, frame_index=idx)
                if frame_callback:
                    frame_callback(res)
                yield res

    @staticmethod
    def _draw_observation_annotation(
        frame: np.ndarray, obs: PipelineObservation
    ) -> None:
        """Draw colored bounding box and plate banner on the frame."""
        det = obs.detection

        # Color coding:
        # Red (0, 0, 255) for Hotlist alert
        # Green (0, 255, 0) for valid recognized plate
        # Yellow (0, 220, 255) for unreadable / low confidence plate
        if obs.is_hotlist_match:
            box_color = (0, 0, 255)  # Red
            status_tag = f"HOTLIST: {obs.plate_number}"
        elif obs.is_valid_plate:
            box_color = (0, 255, 0)  # Green
            status_tag = f"{obs.plate_number} ({obs.ocr_result.confidence:.2f})"
        elif obs.plate_number:
            box_color = (0, 220, 255)  # Yellow
            status_tag = f"{obs.plate_number} ({obs.ocr_result.confidence:.2f})"
        else:
            box_color = (0, 220, 255)  # Yellow
            status_tag = "PLATE (Low OCR)"

        cv2.rectangle(
            frame,
            (det.x1, det.y1),
            (det.x2, det.y2),
            box_color,
            2,
        )

        # Draw banner for text readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 1
        (tw, th), baseline = cv2.getTextSize(status_tag, font, scale, thick)
        ty1 = max(0, det.y1 - th - baseline - 6)
        ty2 = det.y1

        cv2.rectangle(frame, (det.x1, ty1), (det.x1 + tw + 6, ty2), box_color, -1)
        # White text for red boxes, black text for green/yellow boxes
        text_color = (255, 255, 255) if obs.is_hotlist_match else (0, 0, 0)
        cv2.putText(
            frame,
            status_tag,
            (det.x1 + 3, ty2 - baseline - 2),
            font,
            scale,
            text_color,
            thick,
            cv2.LINE_AA,
        )
