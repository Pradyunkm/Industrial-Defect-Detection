"""Shared offline model predictions and review rules (not certified QC)."""
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def predict(image_bgr, model, conf=0.30, iou=0.45, imgsz=320):
    start = time.perf_counter()
    result = model.predict(source=image_bgr, conf=conf, iou=iou, imgsz=imgsz,
                           device='cpu', verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    h, w = image_bgr.shape[:2]
    detections = []
    if result.boxes is not None:
        for b in result.boxes:
            coordinates = [round(float(x), 1) for x in b.xyxy[0].tolist()]
            x1, y1, x2, y2 = coordinates
            detections.append({
                'class': result.names[int(b.cls[0].item())],
                'confidence': round(float(b.conf[0].item()), 3),
                'bbox_xyxy': coordinates,
                'bounding_box_fraction': round(max(0, x2-x1)*max(0, y2-y1)/(w*h), 4),
            })
    return result.plot(), detections, elapsed_ms


def review(detections):
    if not detections:
        return 'UNVERIFIED', 'No prediction exceeded the threshold; this is not proof of a defect-free surface.'
    if any(d['class'] == 'crazing' or d['bounding_box_fraction'] >= .15 for d in detections):
        return 'HIGH REVIEW', 'Demo rule: possible crazing or bounding-box coverage at least 15%; seek human review.'
    if any(d['bounding_box_fraction'] >= .05 for d in detections):
        return 'MEDIUM REVIEW', 'Demo rule: bounding-box coverage at least 5%; seek human review.'
    return 'REVIEW', 'Possible surface defect; human review recommended.'


def write_png(path, bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f'Could not save image: {path}')
