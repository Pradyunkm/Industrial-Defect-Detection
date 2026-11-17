"""Run a single local inspection and save annotated image plus JSON metadata."""
import argparse
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from common import predict, review, write_png

p = argparse.ArgumentParser()
p.add_argument('--model', required=True, help='NCNN exported model directory')
p.add_argument('--image', required=True, help='Existing local input image')
p.add_argument('--output', default='logs/annotated/pi_output.png')
p.add_argument('--conf', type=float, default=.30)
p.add_argument('--imgsz', type=int, default=320)
a = p.parse_args()
image = cv2.imread(a.image)
if image is None:
    p.error(f'Image cannot be opened: {a.image}')
if not Path(a.model).exists():
    p.error(f'Model not found: {a.model}')
model = YOLO(a.model, task='detect')
annotated, boxes, elapsed_ms = predict(image, model, conf=a.conf, imgsz=a.imgsz)
priority, recommendation = review(boxes)
write_png(a.output, annotated)
report = dict(input=a.image, model=a.model, output=a.output, detections=boxes,
              detection_count=len(boxes), review_priority=priority,
              recommendation=recommendation, elapsed_ms=round(elapsed_ms, 1),
              note='Prototype only. No detections do not prove a defect-free surface.')
Path(a.output).with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
