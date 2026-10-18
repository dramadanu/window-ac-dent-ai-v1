"""Draw / confirm the condenser bounding box on each raw image. Output: dataset_raw/boxes.json

For every image the heuristic detector proposes a YELLOW box. You can:
  Enter / Space / y : accept the box on screen (yellow becomes confirmed)
  drag with mouse   : draw your own box instead (replaces the proposal)
  x                 : no condenser in this image (stored as a negative example)
  s                 : skip (leave unlabeled)       b : back to previous image
  q                 : save and quit
Already-labeled images are skipped unless you pass --relabel.
"""
import argparse
import json
from pathlib import Path

import cv2

from detector import HeuristicDetector

RAW = Path("dataset_raw")
BOXES = RAW / "boxes.json"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_W, MAX_H = 1200, 760


def list_images():
    out = []
    for label in ("good", "dent"):
        folder = RAW / label
        if folder.exists():
            out += [p for p in sorted(folder.iterdir()) if p.suffix.lower() in EXT]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relabel", action="store_true", help="also revisit images that already have a label")
    args = ap.parse_args()

    boxes = json.loads(BOXES.read_text()) if BOXES.exists() else {}
    paths = [p for p in list_images() if args.relabel or p.relative_to(RAW).as_posix() not in boxes]
    if not paths:
        print("Nothing to label. Use --relabel to revisit labeled images.")
        return
    heur = HeuristicDetector()
    win = "Label condenser box"
    cv2.namedWindow(win)
    ui = {"drag": False, "p0": None, "p1": None, "box": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ui.update(drag=True, p0=(x, y), p1=(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and ui["drag"]:
            ui["p1"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and ui["drag"]:
            ui["drag"] = False
            (xa, ya), (xb, yb) = ui["p0"], (x, y)
            if abs(xb - xa) > 10 and abs(yb - ya) > 10:
                ui["box"] = (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb), True)  # display coords, confirmed
    cv2.setMouseCallback(win, on_mouse)

    i = 0
    while 0 <= i < len(paths):
        path = paths[i]
        rel = path.relative_to(RAW).as_posix()
        img = cv2.imread(str(path))
        if img is None:
            i += 1
            continue
        h, w = img.shape[:2]
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        disp = cv2.resize(img, (int(w * scale), int(h * scale)))
        if isinstance(boxes.get(rel), list):
            b = boxes[rel]
            ui["box"] = tuple(int(v * scale) for v in b) + (True,)
        else:
            d = heur.detect(img)
            ui["box"] = (tuple(int(v * scale) for v in d.box) + (False,)) if d else None
        action = None
        while action is None:
            view = disp.copy()
            if ui["drag"]:
                cv2.rectangle(view, ui["p0"], ui["p1"], (255, 255, 0), 2)
            elif ui["box"]:
                x1, y1, x2, y2, confirmed = ui["box"]
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0) if confirmed else (0, 220, 255), 3)
            cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(view, f"[{i + 1}/{len(paths)}] {rel} | Enter=accept  drag=redraw  x=no condenser  s=skip  b=back  q=quit",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(win, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32, ord("y")) and ui["box"]:
                x1, y1, x2, y2, _ = ui["box"]
                boxes[rel] = [int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)]
                action = "next"
            elif key == ord("x"):
                boxes[rel] = None
                action = "next"
            elif key == ord("s"):
                action = "next"
            elif key == ord("b"):
                action = "back"
            elif key == ord("q"):
                action = "quit"
        BOXES.write_text(json.dumps(boxes, indent=1))
        if action == "quit":
            break
        i = i + 1 if action == "next" else max(0, i - 1)
    cv2.destroyAllWindows()
    n_box = sum(1 for v in boxes.values() if v)
    print(f"Saved {BOXES}: {n_box} boxes, {len(boxes) - n_box} 'no condenser' entries.")


if __name__ == "__main__":
    main()
