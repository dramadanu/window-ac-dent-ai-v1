# Window AC Condenser Dent AI — V1

Local MVP for classifying the rear side of a window-AC condenser as:
- `good`: no visible dent
- `dent`: visible dent

Stack: Python, OpenCV, PyTorch/MobileNetV3-Small, SQLite, Streamlit. No PLC or cloud required.

## 1. Requirements
- Python 3.10 or 3.11 recommended
- USB webcam connected to the laptop
- Stable, repeatable camera mount and lighting
- Windows/macOS/Linux laptop

## 2. Install
Open a terminal in this folder:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch may need a platform-specific install if pip cannot resolve it. See https://pytorch.org/get-started/locally/

## 3. Capture your own dataset
Run:
```bash
python capture_dataset.py
```
A camera window opens. Aim at the SAME inspection region used in production.
- Press `g` to save a GOOD/no-dent frame.
- Press `d` to save a DENT/NG frame.
- Press `q` to quit.

Images are saved into `dataset_raw/good` and `dataset_raw/dent`.
Capture different units, realistic dent severities, slight lighting variations and normal production variation. Avoid saving many near-identical frames of the same unit.
Do not include personal or confidential material in the frame.

Aim for at least 300–500 distinct images per class for an initial experiment; more diverse examples are better. Include borderline dents and realistic GOOD variation. Verify labels with a quality inspector.

## 4. Split data
Move/copy images into this structure. Keep images from the same physical unit or video burst in only ONE split to avoid leakage:

```text
data/
  train/
    good/
    dent/
  val/
    good/
    dent/
  test/
    good/
    dent/
```

Use roughly 70% train / 15% validation / 15% test. Ensure both classes appear in every split.
`train.py` trains from `data/train` and validates on `data/val`. The `data/test` folder is reserved for a final unbiased evaluation; do not use it to tune the model.

## 5. Train
```bash
python train.py --epochs 12 --batch-size 16
```
The first run may download pretrained MobileNetV3 weights from PyTorch. If offline, pretrained weights may not be available; use an approved network or locally available weights.
Model saved to `models/dent_model.pth`. Training class order is saved in `models/class_names.json`.

## 6. Live inference
```bash
python live_inference.py --camera 0 --interval 1.0 --confidence 0.75
```
- Press `q` in the camera window to quit.
- `--camera 1` can be used if the USB webcam is not device 0.
- `--interval` is the minimum time between logged predictions.
- `--confidence` sets the minimum confidence to label a result as GOOD or NG. Below the threshold, it logs `REVIEW` instead of forcing a decision.
- Press `s` to save a snapshot without creating an extra inspection record.

IMPORTANT: Interval-based inference is a sampling MVP, not guaranteed one-record-per-product tracking. It can inspect the same unit more than once or miss fast-moving units. Before using production counts, validate product spacing, speed, field of view and trigger logic. A photoelectric sensor or robust product-tracking logic may be required for accurate per-unit counting.

## 7. Dashboard
In a second terminal:
```bash
streamlit run dashboard.py
```
Dashboard shows records stored in `data/inspection.db`, hourly counts, NG rate and inspection history. Use the refresh button to reload data.

## 8. Database and image evidence
- SQLite: `data/inspection.db`
- Saved snapshots: `logs/snapshots/`
- Export CSV from the dashboard.

## 9. Validation before production
1. Fix camera angle, focus, distance and lighting.
2. Test on held-out units, not training images.
3. Review confusion matrix and especially NG recall (missed dents are costly).
4. Test across shifts, models, suppliers, surface conditions and dent severity.
5. Define an acceptable confidence threshold with Quality.
6. Keep an inspector as final decision-maker until performance is validated.
7. Do not use the model as a safety interlock or automatic reject system in V1.

## Troubleshooting
- Camera unavailable: close other apps using it; try `--camera 1`.
- `models/dent_model.pth` missing: capture, organize data, then train.
- Predictions are poor: improve lighting and labels, capture more varied examples, check class balance and train/test separation.
- Very slow inference: reduce camera resolution or use a smaller image batch (the script already predicts one image at a time).
