"""Per-unit state machine, counters and on-screen overlay (pure OpenCV/NumPy, no torch needed).

States shown on screen:
  UNDETECTED  default - no condenser in view
  ANALYZING   condenser found, collecting votes from several frames
  OK / NG / REVIEW   locked result for this unit (counted exactly once)
The unit is released (back to UNDETECTED) after the condenser has been gone for `lost_frames` frames.
"""
from typing import Optional

import cv2
import numpy as np

UNDETECTED, ANALYZING, OK, NG, REVIEW = "UNDETECTED", "ANALYZING", "OK", "NG", "REVIEW"
COLORS = {  # BGR
    UNDETECTED: (150, 150, 150), ANALYZING: (255, 200, 0),
    OK: (60, 200, 60), NG: (50, 50, 235), REVIEW: (0, 170, 255),
}
CLASS_TO_STATUS = {"good": OK, "dent": NG}


class Counters:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = self.ok = self.ng = self.review = 0

    def add(self, status: str):
        self.total += 1
        if status == OK:
            self.ok += 1
        elif status == NG:
            self.ng += 1
        else:
            self.review += 1

    def pct(self, n: int) -> float:
        return 100.0 * n / self.total if self.total else 0.0


class UnitTracker:
    def __init__(self, class_names, confidence=0.75, min_frames=5, vote_frames=10, lost_frames=15):
        self.class_names = list(class_names)
        self.confidence, self.min_frames = confidence, min_frames
        self.vote_frames, self.lost_frames = vote_frames, lost_frames
        self.reset()

    def reset(self):
        self.state = UNDETECTED
        self.box = None           # smoothed box for drawing
        self.det_score = 0.0
        self.present = self.lost = 0
        self.probs = []
        self.result_conf = 0.0
        self.live_text = ""

    def _smooth(self, new):
        if self.box is None:
            self.box = tuple(float(v) for v in new)
        else:
            self.box = tuple(0.6 * o + 0.4 * n for o, n in zip(self.box, new))

    def update(self, detection, classify_fn) -> Optional[dict]:
        """Feed one frame's detection. classify_fn() -> probability vector for the current crop.
        Returns an event dict ONCE when a unit's final decision is made, otherwise None."""
        if detection is None:
            self.lost += 1
            if self.lost >= self.lost_frames:
                self.reset()
            return None
        self.lost = 0
        self.present += 1
        self.det_score = detection.score
        self._smooth(detection.box)
        if self.state in (OK, NG, REVIEW):
            return None                                   # already counted; wait until unit leaves
        self.state = ANALYZING
        if self.present <= self.min_frames:               # let box/focus settle first
            self.live_text = "stabilizing..."
            return None
        probs = np.asarray(classify_fn(), dtype=np.float64)
        self.probs.append(probs)
        self.live_text = f"analyzing {len(self.probs)}/{self.vote_frames}"
        if len(self.probs) < self.vote_frames:
            return None
        mean = np.mean(self.probs, axis=0)
        idx = int(mean.argmax())
        conf = float(mean[idx])
        name = self.class_names[idx]
        self.state = CLASS_TO_STATUS.get(name, REVIEW) if conf >= self.confidence else REVIEW
        self.result_conf = conf
        return {"status": self.state, "prediction": name.upper(), "confidence": conf,
                "det_score": self.det_score, "box": tuple(int(v) for v in self.box),
                "votes": len(self.probs)}


def _panel(frame, x1, y1, x2, y2, alpha=0.55):
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def draw_overlay(frame, tracker: UnitTracker, counters: Counters, detector_name: str, hint: str = ""):
    h, w = frame.shape[:2]
    color = COLORS[tracker.state]
    font = cv2.FONT_HERSHEY_SIMPLEX

    if tracker.box is not None and tracker.state != UNDETECTED:       # detection box
        x1, y1, x2, y2 = [int(v) for v in tracker.box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        if tracker.state in (OK, NG, REVIEW):
            label = f"CONDENSER {tracker.det_score:.0%} | {tracker.state} {tracker.result_conf:.0%}"
        else:
            label = f"CONDENSER {tracker.det_score:.0%} | {tracker.live_text}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        ty = y1 - 8 if y1 - th - 12 > 0 else y1 + th + 12
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 6), color, -1)
        cv2.putText(frame, label, (x1 + 4, ty), font, 0.6, (0, 0, 0), 2)

    _panel(frame, 10, 10, 420, 100)                                   # status panel (top-left)
    cv2.putText(frame, "STATUS", (22, 36), font, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, tracker.state, (22, 78), font, 1.2, color, 3)

    c = counters                                                      # counters (top-right)
    px1 = w - 330
    _panel(frame, px1, 10, w - 10, 190)
    cv2.putText(frame, "COUNT (session)", (px1 + 12, 34), font, 0.55, (200, 200, 200), 1)
    rows = [("TOTAL", c.total, None, (255, 255, 255)), ("OK", c.ok, c.pct(c.ok), COLORS[OK]),
            ("NG", c.ng, c.pct(c.ng), COLORS[NG]), ("REVIEW", c.review, c.pct(c.review), COLORS[REVIEW])]
    for i, (name, n, pct, col) in enumerate(rows):
        y = 66 + i * 32
        cv2.putText(frame, name, (px1 + 12, y), font, 0.7, col, 2)
        cv2.putText(frame, f"{n}", (px1 + 140, y), font, 0.7, col, 2)
        if pct is not None:
            cv2.putText(frame, f"{pct:5.1f}%", (px1 + 205, y), font, 0.7, col, 2)

    cv2.putText(frame, f"detector: {detector_name} | {hint}", (14, h - 14), font, 0.5, (255, 255, 255), 1)
    return frame
