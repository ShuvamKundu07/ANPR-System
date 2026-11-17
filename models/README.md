# Model Weights — Source and License

This directory holds YOLO model weights used for license-plate detection.
**Weight files (*.pt, *.onnx) are excluded from version control via .gitignore.**

---

## Approved Model

| Field            | Value                                                                 |
|------------------|-----------------------------------------------------------------------|
| **Name**         | YOLOv8n License Plate Detector                                        |
| **Source**       | [keremberke/yolov8n-license-plate-detection](https://huggingface.co/keremberke/yolov8n-license-plate-detection) |
| **Version**      | v1.0  (best.pt)                                                       |
| **Input size**   | 640 × 640 pixels                                                      |
| **Classes**      | `license-plate` (single class, index 0)                               |
| **License**      | [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| **Paper / ref**  | Ultralytics YOLOv8 — [docs.ultralytics.com](https://docs.ultralytics.com) |

### How to Obtain

```bash
# Option A — Hugging Face Hub (recommended)
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="keremberke/yolov8n-license-plate-detection",
    filename="best.pt",
    local_dir="models",
)
print(f"Downloaded to: {path}")
EOF
mv models/best.pt models/license_plate_detector.pt

# Option B — Manual download
# 1. Visit https://huggingface.co/keremberke/yolov8n-license-plate-detection
# 2. Download best.pt
# 3. Save as: models/license_plate_detector.pt
```

---

## Notes

- Do not replace this model without updating this README.
- If you use a different model, record its name, source URL, training dataset,
  input size, class list, and license here before continuing.
- If the model's license prohibits commercial use, note this clearly.
