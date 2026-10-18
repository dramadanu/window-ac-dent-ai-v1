"""Stage 1 of the pipeline: find WHERE the condenser is in the camera frame.

Two interchangeable detectors, same interface:
  * HeuristicDetector - no training needed. Looks for the dense, parallel fin pattern.
                        Good bootstrap / label proposer, but can be fooled by other striped scenes.
  * LearnedDetector   - Faster R-CNN (MobileNetV3) fine-tuned on your own boxes
                        (see label_boxes.py + train_detector.py). This is the one to use in production.

load_detector("auto") picks the learned model if models/condenser_detector.pth exists,
otherwise falls back to the heuristic.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in original frame pixels
DETECTOR_PATH = "models/condenser_detector.pth"


@dataclass
class Detection:
    box: Box
    score: float


# ----------------------------------------------------------------------------- heuristic
class HeuristicDetector:
    name = "heuristic"

    def __init__(self, min_area_frac: float = 0.25, min_coherence: float = 0.45):
        self.min_area_frac = min_area_frac
        self.min_coherence = min_coherence

    def detect(self, frame) -> Optional[Detection]:
        h, w = frame.shape[:2]
        s = 640.0 / max(h, w)
        small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        k = max(15, (min(small.shape[:2]) // 18) | 1)
        sx, sy = cv2.blur(gx, (k, k)), cv2.blur(gy, (k, k))
        coherence = np.abs(sx - sy) / (sx + sy + 1e-3)      # fins => edges all run one direction
        energy = (sx + sy) / ((sx + sy).max() + 1e-6)
        score_map = cv2.GaussianBlur(coherence * np.sqrt(energy), (0, 0), 5)
        s8 = (255 * score_map / (score_map.max() + 1e-6)).astype(np.uint8)
        _, mask = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh, _ = stats[i]
        comp = (labels == i)[y:y + bh, x:x + bw]
        cols, rows = np.where(comp.mean(axis=0) > 0.5)[0], np.where(comp.mean(axis=1) > 0.5)[0]
        if len(cols) and len(rows):                          # trim ragged edges
            x0, x1, y0, y1 = x + cols[0], x + cols[-1] + 1, y + rows[0], y + rows[-1] + 1
        else:
            x0, x1, y0, y1 = x, x + bw, y, y + bh
        H, W = mask.shape
        area_frac = (x1 - x0) * (y1 - y0) / float(H * W)
        mean_coh = float(coherence[y0:y1, x0:x1].mean())
        if area_frac < self.min_area_frac or mean_coh < self.min_coherence:
            return None
        box = (int(x0 / s), int(y0 / s), int(x1 / s), int(y1 / s))
        return Detection(clip_box(box, w, h), float(min(1.0, mean_coh / 0.85)))


# ----------------------------------------------------------------------------- learned
def build_detector_model(pretrained: bool, min_size: int = 480, max_size: int = 800):
    """Single place that defines the detector architecture (used by training AND inference)."""
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    if pretrained:
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as W
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=W.DEFAULT, min_size=min_size, max_size=max_size)
    else:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None,
                                                      min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)   # background + condenser
    return model


class LearnedDetector:
    name = "learned"

    def __init__(self, model_path: str = DETECTOR_PATH, score_threshold: float = 0.6):
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = build_detector_model(False, ckpt.get("min_size", 480), ckpt.get("max_size", 800))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)
        self.score_threshold = score_threshold

    def detect(self, frame) -> Optional[Detection]:
        torch = self.torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.no_grad():
            out = self.model([tensor])[0]
        if len(out["scores"]) == 0:
            return None
        best = int(out["scores"].argmax().item())
        score = float(out["scores"][best].item())
        if score < self.score_threshold:
            return None
        x1, y1, x2, y2 = [int(round(v)) for v in out["boxes"][best].tolist()]
        h, w = frame.shape[:2]
        box = clip_box((x1, y1, x2, y2), w, h)
        if box[2] - box[0] < 20 or box[3] - box[1] < 20:
            return None
        return Detection(box, score)


# ----------------------------------------------------------------------------- helpers
def clip_box(box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    return (max(0, min(int(x1), w - 1)), max(0, min(int(y1), h - 1)),
            max(1, min(int(x2), w)), max(1, min(int(y2), h)))


def crop_box(frame, box: Box, pad: float = 0.03):
    """Crop the condenser (plus a small margin). Used identically in dataset prep and live inference."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1, x2, y2 = clip_box((x1 - px, y1 - py, x2 + px, y2 + py), w, h)
    return frame[y1:y2, x1:x2]


def load_detector(kind: str = "auto", model_path: str = DETECTOR_PATH, score_threshold: float = 0.6):
    if kind == "heuristic":
        return HeuristicDetector()
    if kind == "learned":
        if not Path(model_path).exists():
            raise SystemExit(f"Learned detector not found: {model_path}. Run label_boxes.py then train_detector.py.")
        return LearnedDetector(model_path, score_threshold)
    if Path(model_path).exists():
        return LearnedDetector(model_path, score_threshold)
    return HeuristicDetector()
