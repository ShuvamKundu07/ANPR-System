# Model Weights — Source and License

This directory holds YOLO model weights used for license-plate detection.
**Weight files (*.pt, *.onnx) are excluded from version control via .gitignore.**

---

## Approved Model

| Field            | Value                                                                 |
|------------------|-----------------------------------------------------------------------|
| **Name**         | YOLOv8n License Plate Detector                                        |
| **Source**       | [Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8) |
| **File**         | `license_plate_detector.pt`                                           |
| **Input size**   | 640 × 640 pixels                                                      |
| **Classes**      | `license_plate` (single class, index 0)                               |
| **License**      | MIT (repository license — see source repo)                            |
| **Framework**    | Ultralytics YOLOv8 — [docs.ultralytics.com](https://docs.ultralytics.com) |

### How to Obtain

```bash
curl -L -o models/license_plate_detector.pt \
  "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
```

Then verify it loads:

```python
from ultralytics import YOLO
model = YOLO("models/license_plate_detector.pt")
print(model.names)   # {0: 'license_plate'}
```

---

## OCR Models (EasyOCR)

| Field            | Value                                                                 |
|------------------|-----------------------------------------------------------------------|
| **Framework**    | EasyOCR (JaidedAI)                                                    |
| **Detection**    | CRAFT (`craft_mlt_25k.pth`)                                           |
| **Recognition**  | English G2 (`english_g2.pth`)                                         |
| **Storage**      | `models/easyocr/model/`                                               |
| **License**      | Apache 2.0                                                            |
| **Source**       | [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)               |

---

## Notes

- Do not replace this model without updating this README.
- If you use a different model, record its name, source URL, training dataset,
  input size, class list, and license here before continuing.
- If the model's license prohibits commercial use, note this clearly.
