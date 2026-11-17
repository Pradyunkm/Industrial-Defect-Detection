"""Evaluate YOLO NCNN on a HELD-OUT, annotated steel-surface dataset.

Accepts Pascal VOC .xml or YOLO normalized .txt bounding-box labels.
Computes per-class precision/recall/F1 and AP@0.50 IoU (not mAP50-95).
Saves JSON and CSV locally in logs/evaluation/; does not edit any model.
"""
import argparse
import csv
import json
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_CLASSES = (
    "crazing", "inclusion", "patches", "pitted_surface",
    "rolled-in_scale", "scratches",
)


def canonical(name):
    name = str(name).lower().strip().replace(" ", "_")
    name = name.replace("-", "_")
    return name


def iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def parse_voc(path):
    root = ET.parse(path).getroot()
    result = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        box = obj.find("bndbox")
        if class_name is None or box is None:
            raise ValueError(f"Invalid VOC object in {path}")
        result.append({
            "class": canonical(class_name),
            "box": [
                float(box.findtext("xmin")), float(box.findtext("ymin")),
                float(box.findtext("xmax")), float(box.findtext("ymax")),
            ],
        })
    return result


def parse_yolo(path, width, height, class_names):
    result = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        pieces = line.split()
        if len(pieces) != 5:
            raise ValueError(f"{path}:{line_no} expected 5 YOLO label columns")
        index = int(pieces[0])
        if index not in class_names:
            raise ValueError(f"{path}:{line_no} unknown class index {index}")
        cx, cy, w, h = map(float, pieces[1:])
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, w, h)):
            raise ValueError(f"{path}:{line_no} out-of-range normalized box")
        result.append({
            "class": canonical(class_names[index]),
            "box": [
                (cx - w / 2) * width, (cy - h / 2) * height,
                (cx + w / 2) * width, (cy + h / 2) * height,
            ],
        })
    return result


def average_precision(points, total_ground_truth):
    """Precision-envelope area under recall curve, IoU fixed at 0.50."""
    if not total_ground_truth:
        return None
    if not points:
        return 0.0
    tp = fp = 0
    precisions = []
    recalls = []
    for point in points:
        if point:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_ground_truth)
    # Integrate the monotonically interpolated precision-recall envelope.
    precisions = [0.0] + precisions + [0.0]
    recalls = [0.0] + recalls + [1.0]
    for position in range(len(precisions) - 2, -1, -1):
        precisions[position] = max(precisions[position], precisions[position + 1])
    return sum(
        (recalls[position] - recalls[position - 1]) * precisions[position]
        for position in range(1, len(recalls))
        if recalls[position] > recalls[position - 1]
    )


def metrics_for_class(predictions, ground_truth_by_image, class_name, threshold=0.50):
    total_ground_truth = sum(
        sum(item["class"] == class_name for item in annotations)
        for annotations in ground_truth_by_image.values()
    )
    candidates = sorted(
        (prediction for prediction in predictions if prediction["class"] == class_name),
        key=lambda item: item["confidence"], reverse=True,
    )
    used = defaultdict(set)
    outcomes = []
    for prediction in candidates:
        image_id = prediction["image"]
        best_index = -1
        best_iou = -1.0
        for index, gt in enumerate(ground_truth_by_image[image_id]):
            if gt["class"] != class_name or index in used[image_id]:
                continue
            score = iou_xyxy(prediction["box"], gt["box"])
            if score > best_iou:
                best_iou, best_index = score, index
        matched = best_index >= 0 and best_iou >= threshold
        if matched:
            used[image_id].add(best_index)
        outcomes.append(matched)
    tp = sum(outcomes)
    fp = len(outcomes) - tp
    fn = total_ground_truth - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_ground_truth if total_ground_truth else None
    f1 = (2 * precision * recall / (precision + recall)
          if recall is not None and (precision + recall) else
          (0.0 if recall is not None else None))
    return {
        "class": class_name,
        "ground_truth_objects": total_ground_truth,
        "predicted_objects": len(outcomes),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 5),
        "recall": round(recall, 5) if recall is not None else None,
        "f1": round(f1, 5) if f1 is not None else None,
        "ap50": round(average_precision(outcomes, total_ground_truth), 5)
        if total_ground_truth else None,
    }


def label_path_for(image_path, image_root, label_root):
    rel = image_path.relative_to(image_root)
    for suffix in (".xml", ".txt"):
        matching = label_root / rel.with_suffix(suffix)
        if matching.is_file():
            return matching
        flat = label_root / (image_path.stem + suffix)
        if flat.is_file():
            return flat
    raise FileNotFoundError(
        f"No matching .xml or .txt label found for {image_path}. "
        "Use a labeled held-out test set; do not treat missing labels as defect-free."
    )


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="Held-out labeled test image folder")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Matching Pascal VOC .xml or YOLO .txt annotation folder")
    parser.add_argument("--model", type=Path,
                        default=Path("models/exported/best_ncnn_model"))
    parser.add_argument("--output", type=Path, default=Path("logs/evaluation"))
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Small confidence cutoff needed for AP50 curves")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional small smoke-test subset. 0 = entire test set")
    args = parser.parse_args()

    if not args.images.is_dir() or not args.labels.is_dir():
        parser.error("Both --images and --labels must be existing folders")
    if not args.model.is_dir():
        parser.error("NCNN model directory missing")
    images = sorted(
        image for image in args.images.rglob("*")
        if image.is_file() and image.suffix.lower() in IMAGE_EXTS
    )
    if args.limit:
        images = images[:args.limit]
    if not images:
        parser.error("No test images were found")

    # Validate all image/annotation pairs before loading the expensive model.
    pairs = [
        (image, label_path_for(image, args.images, args.labels))
        for image in images
    ]
    print(f"Evaluating {len(pairs)} annotated images (IoU = 0.50)", flush=True)
    print("WARNING: Use a true held-out set not used in YOLO training.", flush=True)

    import cv2
    from ultralytics import YOLO

    model = YOLO(str(args.model), task="detect")
    class_names = {
        int(index): canonical(name) for index, name in model.names.items()
    }
    expected = {canonical(name) for name in EXPECTED_CLASSES}
    if not expected.issubset(set(class_names.values())):
        print("WARNING: Model does not contain all expected NEU-DET class names.")
        print("Model names:", class_names, flush=True)

    ground_truth = {}
    predictions = []
    inference_ms = []
    for number, (image, label) in enumerate(pairs, start=1):
        frame = cv2.imread(str(image))
        if frame is None:
            raise RuntimeError(f"Cannot decode image: {image}")
        height, width = frame.shape[:2]
        annotations = (
            parse_voc(label) if label.suffix.lower() == ".xml"
            else parse_yolo(label, width, height, class_names)
        )
        unexpected = {item["class"] for item in annotations} - set(class_names.values())
        if unexpected:
            raise ValueError(f"Unknown ground-truth classes in {label}: {sorted(unexpected)}")
        key = image.relative_to(args.images).as_posix()
        ground_truth[key] = annotations

        start = time.perf_counter()
        result = model.predict(
            source=frame, conf=args.conf, imgsz=args.imgsz,
            device="cpu", verbose=False
        )[0]
        inference_ms.append((time.perf_counter() - start) * 1000)
        if result.boxes is not None:
            for box in result.boxes:
                predictions.append({
                    "image": key,
                    "class": class_names[int(box.cls[0].item())],
                    "confidence": float(box.conf[0].item()),
                    "box": [float(value) for value in box.xyxy[0].tolist()],
                })
        if number % 10 == 0 or number == len(pairs):
            print(f"Processed {number}/{len(pairs)} images", flush=True)

    rows = [
        metrics_for_class(predictions, ground_truth, canonical(name))
        for name in EXPECTED_CLASSES
    ]
    populated_aps = [row["ap50"] for row in rows if row["ap50"] is not None]
    all_classes_present = len(populated_aps) == len(EXPECTED_CLASSES)
    aggregate = {
        "evaluated_images": len(pairs),
        "total_ground_truth_objects": sum(row["ground_truth_objects"] for row in rows),
        "model": str(args.model),
        "confidence_cutoff": args.conf,
        "iou_threshold": 0.50,
        "imgsz": args.imgsz,
        "average_yolo_latency_ms": round(statistics.mean(inference_ms), 1),
        "median_yolo_latency_ms": round(statistics.median(inference_ms), 1),
        "mAP50_available_classes": round(statistics.mean(populated_aps), 5) if populated_aps else None,
        "mAP50_all_six_classes": (
            round(statistics.mean(populated_aps), 5) if all_classes_present else None
        ),
        "all_six_classes_represented": all_classes_present,
        "mAP50_95": "not measured by this evaluation script",
        "per_class": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_csv(args.output / "per_class_metrics.csv", rows, list(rows[0]))
    print("\nEvaluation results (AP@0.50 IoU):", flush=True)
    for row in rows:
        print(
            f"  {row['class']}: GT {row['ground_truth_objects']}, "
            f"TP {row['tp']}, FP {row['fp']}, FN {row['fn']}, "
            f"AP50 {row['ap50'] if row['ap50'] is not None else 'N/A'}",
            flush=True,
        )
    print(f"mAP50 over represented classes: {aggregate['mAP50_available_classes']}")
    if not all_classes_present:
        print("No overall six-class mAP50 claim: some classes have no annotated examples.")
    print(f"Results saved under {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
