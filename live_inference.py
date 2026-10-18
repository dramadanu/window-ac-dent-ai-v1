"""Live inspection V2:  camera -> (1) find condenser box -> (2) classify the crop -> OK / NG -> count once per unit."""
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from common import init_db, log_inspection, counts_today, SNAPSHOT_DIR
from detector import load_detector, crop_box
from inspector import UnitTracker, Counters, draw_overlay, OK, NG, REVIEW, UNDETECTED


def load_classifier(model_path, device):
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
    p.add_argument("--model", default="models/dent_model.pth", help="dent classifier")
    p.add_argument("--detector", default="auto", choices=["auto", "heuristic", "learned"])
    p.add_argument("--detector-model", default="models/condenser_detector.pth")
    p.add_argument("--det-threshold", type=float, default=0.6, help="min detector confidence (learned detector)")
    p.add_argument("--confidence", type=float, default=0.75, help="min classifier confidence, else REVIEW")
    p.add_argument("--min-frames", type=int, default=5, help="frames to wait after condenser appears")
    p.add_argument("--vote-frames", type=int, default=10, help="frames averaged for the final decision")
    p.add_argument("--lost-frames", type=int, default=15, help="frames without condenser before the unit is released")
    p.add_argument("--detect-every", type=int, default=1, help="run the detector every N frames (raise if slow)")
    p.add_argument("--resume", action="store_true", help="start counters from today's logged totals")
    args = p.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Classifier not found: {args.model}. Run prepare_dataset.py and train.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, tf = load_classifier(args.model, device)
    detector = load_detector(args.detector, args.detector_model, args.det_threshold)
    init_db()

    tracker = UnitTracker(classes, args.confidence, args.min_frames, args.vote_frames, args.lost_frames)
    counters = Counters()
    if args.resume:
        for status, n in counts_today().items():
            counters.total += n
            if status == OK:
                counters.ok += n
            elif status == NG:
                counters.ng += n
            elif status == REVIEW:
                counters.review += n

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Try --camera 1 or close other camera apps.")

    print(f"Detector: {detector.name} | classes: {classes}")
    print("q=quit | s=snapshot | n=re-inspect current unit | r=reset session counters")
    frame_idx, last_det = 0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame.")
                break
            if frame_idx % max(args.detect_every, 1) == 0:
                last_det = detector.detect(frame)
            frame_idx += 1
            det = last_det

            def classify():
                crop = crop_box(frame, det.box)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
                with torch.no_grad():
                    return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

            event = tracker.update(det, classify)
            if event:                                   # unit finished: count + log exactly once
                counters.add(event["status"])
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                evidence = frame.copy()
                x1, y1, x2, y2 = event["box"]
                cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(evidence, f"{event['status']} {event['confidence']:.0%}", (x1 + 8, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                img_path = SNAPSHOT_DIR / f"{stamp}_{event['status'].lower()}.jpg"
                cv2.imwrite(str(img_path), evidence, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                log_inspection(event["prediction"], event["confidence"], event["status"], img_path,
                               source="webcam_unit", det_score=event["det_score"],
                               box=event["box"], votes=event["votes"])
                print(f"[{counters.total}] {event['status']} ({event['confidence']:.1%}) "
                      f"| OK={counters.ok} NG={counters.ng} REVIEW={counters.review}")

            view = frame.copy()
            draw_overlay(view, tracker, counters, detector.name, "q quit | s snap | n re-inspect | r reset")
            cv2.imshow("Window AC Condenser - AI Visual Inspection V2", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                path = SNAPSHOT_DIR / f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(str(path), view)
                print(f"Snapshot saved: {path}")
            elif key == ord("n"):
                tracker.reset()
            elif key == ord("r"):
                counters.reset()
                print("Session counters reset.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
