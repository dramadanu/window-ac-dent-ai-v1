import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn

from common import init_db, log_inspection, SNAPSHOT_DIR

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    classes = checkpoint["class_names"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, classes, tf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval", type=float, default=1.0, help="Minimum seconds between logged predictions")
    p.add_argument("--confidence", type=float, default=0.75)
    p.add_argument("--model", default="models/dent_model.pth")
    args = p.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}. Follow README to capture data and train first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, tf = load_model(args.model, device)
    init_db()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Try --camera 1 or close other camera apps.")
    last_log = 0.0
    last_result = ("WAITING", 0.0, "WAITING")
    print("Live inference running. q=quit, s=save snapshot (without logging another record).")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame.")
                break
            now = time.monotonic()
            if now - last_log >= max(args.interval, 0.1):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0]
                idx = int(probs.argmax().item())
                prediction = classes[idx]
                confidence = float(probs[idx].item())
                status = prediction.upper() if confidence >= args.confidence else "REVIEW"
                last_result = (prediction.upper(), confidence, status)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_path = SNAPSHOT_DIR / f"{stamp}_{status.lower()}.jpg"
                cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                log_inspection(prediction.upper(), confidence, status, img_path)
                last_log = now

            pred, conf, status = last_result
            color = (40, 200, 40) if status == "GOOD" else (30, 30, 230) if status in ("DENT", "NG") else (0, 190, 255)
            cv2.rectangle(frame, (10, 10), (620, 115), (20, 20, 20), -1)
            cv2.putText(frame, f"Prediction: {pred} | Decision: {status}", (25, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Confidence: {conf:.1%} | Log interval: {args.interval:.1f}s",
                        (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
            cv2.putText(frame, "q=quit | s=snapshot", (20, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Window AC Condenser - AI Visual Inspection V1", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = SNAPSHOT_DIR / f"manual_{stamp}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Snapshot saved: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
