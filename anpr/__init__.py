"""
anpr package — Automatic License Plate Recognition POC
"""

from anpr.alerts import AlertManager
from anpr.database import Database
from anpr.detector import Detection, LicensePlateDetector
from anpr.hotlist import HotlistMatcher
from anpr.ocr import OCRResult, PlateOCR
from anpr.pipeline import ANPRPipeline, PipelineObservation, PipelineResult
from anpr.plate_utils import normalize_plate, validate_plate
from anpr.tracking import DuplicateTracker, VehicleJourneyTracker
from anpr.video import frame_generator, get_video_properties, open_source

__all__ = [
    "AlertManager",
    "ANPRPipeline",
    "Database",
    "Detection",
    "DuplicateTracker",
    "HotlistMatcher",
    "LicensePlateDetector",
    "OCRResult",
    "PipelineObservation",
    "PipelineResult",
    "PlateOCR",
    "VehicleJourneyTracker",
    "frame_generator",
    "get_video_properties",
    "normalize_plate",
    "open_source",
    "validate_plate",
]
