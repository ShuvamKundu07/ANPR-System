"""
anpr/ocr.py
───────────
Phase 4 — Optical Character Recognition with EasyOCR.

Provides:
  • OCRResult dataclass: raw text, confidence score, and token details.
  • PlateOCR: reusable OCR engine wrapping EasyOCR.

Initializes the reader once, reuses weights across frames, and handles
empty, low-confidence, or unreadable crops safely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import cv2
import easyocr
import numpy as np

from anpr.config import (
    EASYOCR_MODEL_DIR,
    EASYOCR_USER_NETWORK_DIR,
    OCR_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRResult:
    """Represents character recognition output for a plate crop."""

    raw_text: str = ""
    confidence: float = 0.0
    tokens: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.raw_text.strip()


class PlateOCR:
    """Wraps EasyOCR Reader for plate recognition, reusing loaded weights."""

    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        gpu: bool = False,
        model_storage_dir: Union[str, Path, None] = None,
        user_network_dir: Union[str, Path, None] = None,
        confidence_threshold: float | None = None,
        download_enabled: bool = True,
    ) -> None:
        """Initialize EasyOCR reader once.

        Args:
            languages: Tuple/list of language codes (default: ('en',)).
            gpu: Whether to enable GPU inference (default False for CPU portability).
            model_storage_dir: Directory where model weights are stored.
            user_network_dir: Directory for custom networks.
            confidence_threshold: Default minimum confidence for recognized fragments.
            download_enabled: Whether to allow EasyOCR to download models if missing.
        """
        self.model_storage_dir = (
            Path(model_storage_dir) if model_storage_dir else EASYOCR_MODEL_DIR
        )
        self.user_network_dir = (
            Path(user_network_dir) if user_network_dir else EASYOCR_USER_NETWORK_DIR
        )
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else OCR_CONFIDENCE_THRESHOLD
        )

        self.model_storage_dir.mkdir(parents=True, exist_ok=True)
        self.user_network_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing EasyOCR Reader (gpu=%s, model_dir=%s, download_enabled=%s)",
            gpu,
            self.model_storage_dir,
            download_enabled,
        )
        self.reader = easyocr.Reader(
            lang_list=list(languages),
            gpu=gpu,
            model_storage_directory=str(self.model_storage_dir),
            user_network_directory=str(self.user_network_dir),
            download_enabled=download_enabled,
        )
        logger.info("EasyOCR Reader initialized successfully.")

    def read_plate(
        self,
        image: Union[np.ndarray, str, Path],
        min_confidence: float | None = None,
    ) -> OCRResult:
        """Run OCR on a cropped license-plate image.

        Args:
            image: Plate crop as NumPy BGR array, or file path.
            min_confidence: Optional confidence cutoff for individual fragments.

        Returns:
            OCRResult with raw recognised text, average confidence, and token pairs.
        """
        conf_cutoff = (
            min_confidence
            if min_confidence is not None
            else self.confidence_threshold
        )

        # Handle path input
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.exists():
                logger.warning("Image path does not exist: %s", p)
                return OCRResult()
            img_arr = cv2.imread(str(p))
            if img_arr is None:
                logger.warning("Could not read image from file: %s", p)
                return OCRResult()
        elif isinstance(image, np.ndarray):
            img_arr = image
        else:
            logger.warning("Unsupported image type: %s", type(image))
            return OCRResult()

        if img_arr.size == 0 or img_arr.ndim < 2:
            logger.debug("Received empty image array for OCR.")
            return OCRResult()

        # Scale small crops up to a minimum height for CRAFT detector readability
        h, w = img_arr.shape[:2]
        if h < 60 and w > 0:
            scale = max(2.0, 70.0 / h)
            ocr_input = cv2.resize(
                img_arr,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            ocr_input = img_arr

        try:
            # EasyOCR readtext with alphanumeric plate allowlist
            raw_results = self.reader.readtext(
                ocr_input,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
                detail=1,
            )
        except Exception:
            try:
                raw_results = self.reader.readtext(ocr_input, detail=1)
            except Exception as err:
                logger.error("EasyOCR inference failed: %s", err, exc_info=True)
                return OCRResult()

        if not raw_results:
            return OCRResult()

        # Sort tokens top-to-bottom, then left-to-right based on bounding box
        # bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        def _sort_key(item: tuple) -> tuple[float, float]:
            bbox = item[0]
            top_y = float(bbox[0][1])
            left_x = float(bbox[0][0])
            return (top_y, left_x)

        sorted_results = sorted(raw_results, key=_sort_key)

        accepted_tokens: list[tuple[str, float]] = []
        confidences: list[float] = []

        for _bbox, text, conf in sorted_results:
            clean_txt = str(text).strip()
            score = float(conf)
            if not clean_txt:
                continue
            if score >= conf_cutoff:
                accepted_tokens.append((clean_txt, score))
                confidences.append(score)
            else:
                logger.debug(
                    "Token %r discarded below confidence cutoff (%.2f < %.2f)",
                    clean_txt,
                    score,
                    conf_cutoff,
                )

        if not accepted_tokens:
            return OCRResult()

        combined_text = " ".join(t[0] for t in accepted_tokens)
        mean_confidence = float(np.mean(confidences))

        return OCRResult(
            raw_text=combined_text,
            confidence=round(mean_confidence, 4),
            tokens=tuple(accepted_tokens),
        )
