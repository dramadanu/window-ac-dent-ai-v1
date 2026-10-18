import cv2
from pathlib import Path
from datetime import datetime

ROOT = Path("dataset_raw")
for label in ("good", "dent", "background"):
    (ROOT / label).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam. Close other camera apps or edit camera index in capture_dataset.py.")

KEYS = {ord("g"): "good", ord("d"): "dent", ord("b"): "background"}
print("g=GOOD (condenser, no dent) | d=DENT/NG (condenser) | b=BACKGROUND (NO condenser in view) | q=quit")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame read failed.")
        break
    preview = frame.copy()
    cv2.putText(preview, "g: GOOD | d: DENT/NG | b: BACKGROUND (no condenser) | q: quit", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 255), 2)
    cv2.imshow("Dataset Capture - Window AC Condenser", preview)
    key = cv2.waitKey(1) & 0xFF
    if key in KEYS:
        label = KEYS[key]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = ROOT / label / f"{label}_{stamp}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Saved: {path}")
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
