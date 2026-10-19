"""Detect objects with YOLOv8 and replace one of them: inpaint its box and paste another image there.

Example:
    python -m vision.detection.replace_object --image street.jpg --replacement sticker.png --target car --index 5
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def label_detections(boxes, class_names) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Name detections as car, car_2, car_3, ... in detection order."""
    counts, labeled = Counter(), []
    for name, box in zip(class_names, boxes):
        counts[name] += 1
        labeled.append((name if counts[name] == 1 else f"{name}_{counts[name]}", tuple(map(int, box))))
    return labeled


def paste_with_alpha(background: np.ndarray, overlay: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    overlay = cv2.resize(overlay, (x2 - x1, y2 - y1))
    result = background.copy()
    if overlay.ndim == 3 and overlay.shape[2] == 4:
        alpha = overlay[:, :, 3:4] / 255.0
        result[y1:y2, x1:x2] = (alpha * overlay[:, :, :3] + (1 - alpha) * result[y1:y2, x1:x2]).astype(np.uint8)
    else:
        result[y1:y2, x1:x2] = overlay[:, :, :3]
    return result


def replace_object(image: np.ndarray, box: tuple[int, int, int, int], replacement: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    background = cv2.inpaint(image, mask, 10, cv2.INPAINT_TELEA)
    return paste_with_alpha(background, replacement, box)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--target", default="car", help="COCO class name of the object to replace")
    parser.add_argument("--index", type=int, default=1, help="which object of that class (1-based)")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--out", type=Path, default=Path("results/detection"))
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    image = cv2.imread(str(args.image))
    result = model(image)[0]
    names = [model.names[int(c)] for c in result.boxes.cls]
    detections = label_detections(result.boxes.xyxy.tolist(), names)
    print(f"Detected {len(detections)} objects: {dict(Counter(names))}")

    annotated = result.plot()
    for label, (x1, y1, _, _) in detections:
        cv2.putText(annotated, label, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    targets = [box for label, box in detections if label.split("_")[0] == args.target]
    if len(targets) < args.index:
        raise SystemExit(f"Only {len(targets)} objects of class '{args.target}' found")
    replaced = replace_object(image, targets[args.index - 1], cv2.imread(str(args.replacement), cv2.IMREAD_UNCHANGED))

    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "detections.jpg"), annotated)
    cv2.imwrite(str(args.out / "replaced.jpg"), replaced)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
