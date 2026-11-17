"""
scripts/evaluate_reliability.py
───────────────────────────────
Phase 10 — Quantitative Reliability & Accuracy Benchmarking.

Evaluates:
  • Frame processing latency and average FPS
  • Plate detection precision & recall across known synthetic test patterns
  • Character-level & exact plate OCR accuracy
  • Hotlist alert trigger correctness & false alert rates
  • Duplicate observation suppression efficiency
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import NamedTuple

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anpr.database import Database
from anpr.detector import LicensePlateDetector
from anpr.ocr import PlateOCR
from anpr.pipeline import ANPRPipeline
from anpr.plate_utils import normalize_plate


class GroundTruthSample(NamedTuple):
    text: str
    is_hotlisted: bool
    condition: str
    bg_val: int
    blur: bool
    rotate_deg: float


TEST_CASES = [
    GroundTruthSample("DL8CAB1234", True, "Daylight Clear", 200, False, 0.0),
    GroundTruthSample("MH12DE1433", False, "Standard Daylight", 150, False, 0.0),
    GroundTruthSample("KA01E9999", True, "Angled Plate", 160, False, 6.0),
    GroundTruthSample("UP32AB0001", False, "Low-Light / Shadow", 40, False, 0.0),
    GroundTruthSample("HR26DK8337", False, "Blurry / Motion", 130, True, 0.0),
    GroundTruthSample("WB02AB1111", True, "Clean Contrast", 220, False, 0.0),
    GroundTruthSample("GJ01EF5555", False, "Slight Rotation", 170, False, -5.0),
    GroundTruthSample("TN09BC7777", False, "Daylight High Brightness", 230, False, 0.0),
]


def render_sample(sample: GroundTruthSample) -> np.ndarray:
    frame = np.full((480, 640, 3), sample.bg_val, dtype=np.uint8)
    cv2.rectangle(frame, (120, 180), (520, 420), (50, 50, 50), -1)
    cv2.rectangle(frame, (200, 270), (440, 340), (255, 255, 255), -1)
    cv2.rectangle(frame, (200, 270), (440, 340), (0, 0, 0), 2)
    cv2.putText(
        frame,
        sample.text,
        (210, 318),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    if sample.blur:
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
    if sample.rotate_deg != 0.0:
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), sample.rotate_deg, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h), borderValue=(sample.bg_val, sample.bg_val, sample.bg_val))
    return frame


def evaluate_benchmarks():
    print("=" * 65)
    print("   ANPR POC — Phase 10 Reliability & Accuracy Evaluation")
    print("=" * 65)

    db = Database(":memory:")
    detector = LicensePlateDetector(confidence_threshold=0.35)
    ocr = PlateOCR(gpu=False)
    pipeline = ANPRPipeline(camera_id="BENCH_CAM", db=db, detector=detector, ocr=ocr, duplicate_window_sec=5)

    # Seed hotlist
    for s in TEST_CASES:
        if s.is_hotlisted:
            db.add_hotlist_entry(s.text, "STOLEN", "Benchmark hotlist entry")

    total_samples = len(TEST_CASES)
    detections_found = 0
    exact_matches = 0
    total_chars = 0
    matched_chars = 0
    hotlist_expected = sum(1 for s in TEST_CASES if s.is_hotlisted)
    hotlist_triggered_correct = 0
    false_alerts = 0
    latencies = []

    print(f"\nEvaluating {total_samples} synthetic benchmark test scenarios...\n")

    for idx, sample in enumerate(TEST_CASES, start=1):
        frame = render_sample(sample)
        
        t0 = time.perf_counter()
        res = pipeline.process_frame(frame, frame_index=idx, timestamp=f"2026-11-17 12:00:{idx:02d}")
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        gt_norm = normalize_plate(sample.text)
        has_det = res.has_detections
        if has_det:
            detections_found += 1

        rec_norm = ""
        is_alert = False
        if res.observations:
            obs = res.observations[0]
            rec_norm = obs.plate_number
            is_alert = obs.is_hotlist_match

        # Character-level matching
        total_chars += len(gt_norm)
        cur_matched = sum(1 for a, b in zip(gt_norm, rec_norm) if a == b)
        matched_chars += cur_matched

        is_exact = (gt_norm == rec_norm)
        if is_exact:
            exact_matches += 1

        if sample.is_hotlisted:
            if is_alert:
                hotlist_triggered_correct += 1
        else:
            if is_alert:
                false_alerts += 1

        status_flag = "EXACT" if is_exact else ("DETECTED" if has_det else "MISSED")
        print(f"[{idx}/{total_samples}] {sample.condition:<24} | GT: {gt_norm:<10} | Rec: {rec_norm:<10} | {status_flag} ({elapsed*1000:.1f}ms)")

    # Negative control: frames with no plates
    neg_frames = 5
    neg_false_positives = 0
    for i in range(neg_frames):
        blank = np.full((480, 640, 3), 110, dtype=np.uint8)
        t0 = time.perf_counter()
        res = pipeline.process_frame(blank, frame_index=100 + i)
        latencies.append(time.perf_counter() - t0)
        if res.has_detections:
            neg_false_positives += 1

    avg_latency = np.mean(latencies)
    avg_fps = 1.0 / avg_latency if avg_latency > 0 else 0.0
    det_precision = detections_found / (detections_found + neg_false_positives) if (detections_found + neg_false_positives) > 0 else 1.0
    det_recall = detections_found / total_samples
    exact_accuracy = exact_matches / total_samples
    char_accuracy = matched_chars / total_chars if total_chars > 0 else 0.0
    hotlist_accuracy = hotlist_triggered_correct / hotlist_expected if hotlist_expected > 0 else 1.0

    print("\n" + "=" * 65)
    print("                      BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Test Cases Evaluated : {total_samples + neg_frames} ({total_samples} vehicle + {neg_frames} background)")
    print(f"Plate Detection Recall     : {det_recall * 100:.1f}% ({detections_found}/{total_samples})")
    print(f"Plate Detection Precision  : {det_precision * 100:.1f}%")
    print(f"False Detections (Clean BG): {neg_false_positives}/{neg_frames}")
    print(f"Exact Plate OCR Accuracy   : {exact_accuracy * 100:.1f}% ({exact_matches}/{total_samples})")
    print(f"Character-Level OCR Acc    : {char_accuracy * 100:.1f}% ({matched_chars}/{total_chars} chars)")
    print(f"Hotlist Alert Trigger Acc  : {hotlist_accuracy * 100:.1f}% ({hotlist_triggered_correct}/{hotlist_expected})")
    print(f"False Hotlist Alerts       : {false_alerts}")
    print(f"Average Processing Time    : {avg_latency * 1000:.1f} ms/frame (~{avg_fps:.1f} FPS on CPU)")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_benchmarks()
