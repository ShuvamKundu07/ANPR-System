"""
scripts/simulate_two_cameras.py
────────────────────────────────
Phase 9 — Two-Camera Movement History Demonstration.

Simulates two independent camera sources processing the SAME video file,
each assigned a different camera ID (CAM001 / CAM002) and a staggered
start timestamp (12 minutes apart), so the resulting SQLite database
contains observations showing the same plate travelling between cameras.

This is a CONTROLLED SIMULATION, not live multi-camera visual tracking.
Plate identity is linked purely by normalized OCR text — not by visual
vehicle re-identification.

Usage (from project root with venv activated):
    python scripts/simulate_two_cameras.py [--video PATH] [--max-frames N]

Outputs:
    • Observations in data/anpr.db tagged with CAM001 / CAM002
    • Plate crops in data/plate_crops/
    • A printed timeline report in the terminal
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Make sure project root is on sys.path when run directly ───────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anpr.config import CAM_ID_1, CAM_ID_2, PROJECT_ROOT as CFG_ROOT
from anpr.database import Database
from anpr.detector import LicensePlateDetector
from anpr.ocr import PlateOCR
from anpr.pipeline import ANPRPipeline
from anpr.plate_utils import normalize_plate
from anpr.video import frame_generator, open_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("simulate_two_cameras")


# ── Default video path ─────────────────────────────────────────────────────────
DEFAULT_VIDEO = str(CFG_ROOT / "data" / "input_videos" / "sample_test.mp4")

# ── Camera simulation parameters ──────────────────────────────────────────────
# CAM001 "sees" the vehicle first; CAM002 sees it 12 minutes later.
CAM001_START = datetime(2026, 11, 17, 10, 30, 0, tzinfo=timezone.utc)
CAM002_START = CAM001_START + timedelta(minutes=12)

# Location labels for the demo display
CAM_LOCATIONS = {
    CAM_ID_1: "Entry Gate — Sector 12",
    CAM_ID_2: "Exit Gate — Sector 14",
}


def _make_pipeline(
    camera_id: str,
    db: Database,
    detector: LicensePlateDetector,
    ocr: PlateOCR,
) -> ANPRPipeline:
    """Create a pipeline instance for one simulated camera."""
    return ANPRPipeline(
        camera_id=camera_id,
        db=db,
        detector=detector,
        ocr=ocr,
        # Use a short duplicate window so different cameras don't suppress each other
        duplicate_window_sec=5,
    )


def simulate_camera(
    pipeline: ANPRPipeline,
    video_path: str,
    start_time: datetime,
    max_frames: int,
    fps: float = 25.0,
) -> list[str]:
    """
    Process video through one simulated camera pipeline.

    Timestamps are synthesized from start_time + frame_index / fps so
    each frame has a realistic timestamp independent of wall-clock time.

    Returns:
        List of normalized plate numbers recognized.
    """
    cam = pipeline.camera_id
    logger.info("=== Simulating %s (location: %s) ===", cam, CAM_LOCATIONS.get(cam, cam))
    logger.info("    Video  : %s", video_path)
    logger.info("    Start  : %s", start_time.strftime("%Y-%m-%d %H:%M:%S UTC"))

    plates_seen: list[str] = []
    frame_count = 0

    try:
        with open_source(video_path) as cap:
            for idx, frame in frame_generator(cap, max_frames=max_frames):
                # Synthesize timestamp: start + elapsed seconds at video FPS
                ts = (start_time + timedelta(seconds=idx / fps)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                result = pipeline.process_frame(frame, frame_index=idx, timestamp=ts)
                frame_count += 1

                for obs in result.observations:
                    if obs.is_valid_plate and not obs.is_duplicate and obs.plate_number:
                        plates_seen.append(obs.plate_number)
                        logger.info(
                            "  [%s] Frame %d | %s | plate=%s | conf=%.2f | db_id=%s",
                            cam,
                            idx,
                            ts,
                            obs.plate_number,
                            obs.ocr_result.confidence,
                            obs.db_record_id,
                        )

    except FileNotFoundError as exc:
        logger.error("Video not found: %s", exc)
    except Exception as exc:
        logger.error("Error during simulation of %s: %s", cam, exc, exc_info=True)

    logger.info(
        "=== %s done | %d frames | %d unique plate records ===",
        cam,
        frame_count,
        len(plates_seen),
    )
    return plates_seen


def print_movement_report(db: Database) -> None:
    """
    Query the database and print a formatted cross-camera movement timeline
    for every plate that appears in more than one camera.
    """
    all_records = db.get_recent_observations(limit=500)
    if not all_records:
        print("\n⚠  No observations in database — nothing to report.")
        return

    # Group by normalized plate
    from collections import defaultdict
    by_plate: dict[str, list[dict]] = defaultdict(list)
    for rec in all_records:
        by_plate[rec["plate_number"]].append(rec)

    # Sort each group chronologically
    for plate in by_plate:
        by_plate[plate].sort(key=lambda r: r["detected_at"])

    print("\n" + "=" * 70)
    print("  TWO-CAMERA MOVEMENT HISTORY REPORT")
    print("  Plate identity linked by OCR — confirm independently.")
    print("=" * 70)

    multi_cam = {p: recs for p, recs in by_plate.items() if len({r["camera_id"] for r in recs}) > 1}
    single_cam = {p: recs for p, recs in by_plate.items() if p not in multi_cam}

    if multi_cam:
        print(f"\n📍 Plates seen on MULTIPLE cameras ({len(multi_cam)}):\n")
        for plate, recs in sorted(multi_cam.items()):
            cams = sorted({r["camera_id"] for r in recs})
            print(f"  Plate: {plate}  |  Cameras: {', '.join(cams)}")
            for i, rec in enumerate(recs, 1):
                loc = CAM_LOCATIONS.get(rec["camera_id"], rec["camera_id"])
                print(
                    f"    {i}. [{rec['camera_id']}] {loc}  —  {rec['detected_at']}"
                    f"  (OCR conf: {rec['ocr_confidence']:.2f})"
                )
            print()
    else:
        print("\n  No plates were seen on more than one camera in this run.")

    if single_cam:
        print(f"  Plates seen on only ONE camera: {len(single_cam)}")
        for plate, recs in sorted(single_cam.items()):
            cam_ids = {r["camera_id"] for r in recs}
            print(f"    {plate}  →  {', '.join(sorted(cam_ids))}  ({len(recs)} observation(s))")

    print("\n" + "=" * 70)
    print(f"  Total observations: {len(all_records)}")
    print(f"  Total unique plates: {len(by_plate)}")
    print(f"  Plates with cross-camera history: {len(multi_cam)}")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate two-camera ANPR and show movement history."
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help="Path to input video (default: data/input_videos/sample_test.mp4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Max frames per camera to process (default: 150)",
    )
    args = parser.parse_args()

    video_path = str(Path(args.video).resolve())
    if not Path(video_path).exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    logger.info("Initializing shared models (loaded once, reused for both cameras)…")

    # Shared heavy models — loaded once and reused for both cameras
    db = Database()
    detector = LicensePlateDetector()
    ocr = PlateOCR(gpu=False)

    # Pipeline per camera
    pipeline_cam1 = _make_pipeline(CAM_ID_1, db, detector, ocr)
    pipeline_cam2 = _make_pipeline(CAM_ID_2, db, detector, ocr)

    # --- CAM001 pass ---
    plates_cam1 = simulate_camera(
        pipeline_cam1,
        video_path,
        start_time=CAM001_START,
        max_frames=args.max_frames,
    )

    # --- CAM002 pass (12 minutes later) ---
    plates_cam2 = simulate_camera(
        pipeline_cam2,
        video_path,
        start_time=CAM002_START,
        max_frames=args.max_frames,
    )

    # --- Print movement report ---
    print_movement_report(db)

    # Summarize
    all_seen = set(plates_cam1) | set(plates_cam2)
    shared = set(plates_cam1) & set(plates_cam2)
    print(f"  CAM001 unique plates : {len(set(plates_cam1))}")
    print(f"  CAM002 unique plates : {len(set(plates_cam2))}")
    print(f"  Plates on BOTH cams  : {len(shared)}")
    if shared:
        print(f"  → {', '.join(sorted(shared))}")
    print()


if __name__ == "__main__":
    main()
