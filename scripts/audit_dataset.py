"""Read-only NEU-DET dataset inventory: find real image/annotation pairs and split evidence.

Never creates a random test split or edits user datasets. This cannot prove
train/test independence without records of how the NCNN model was trained.
"""
import argparse
import collections
import json
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}
EXPECTED = {
    "crazing", "inclusion", "patches", "pitted_surface",
    "rolled_in_scale", "scratches",
}


def split_for(path):
    for name in path.parts:
        if name.lower() in SPLITS:
            return SPLITS[name.lower()]
    return "unspecified"


def paired_label(image, root, by_stem):
    """Return (label, ambiguity) rather than guessing on duplicate stems."""
    candidates = []
    relative = image.relative_to(root)
    parts = list(relative.parts)
    for index, part in enumerate(parts[:-1]):
        if part.lower() in {"images", "image", "imgs"}:
            for new in ("labels", "LABELS", "annotations", "ANNOTATIONS", "Annotations"):
                new_parts = parts.copy()
                new_parts[index] = new
                base = root.joinpath(*new_parts).with_suffix("")
                candidates.extend(base.with_suffix(ext) for ext in (".xml", ".txt"))
    for ext in (".xml", ".txt"):
        candidates.append(image.with_suffix(ext))
        for subdir in ("ANNOTATIONS", "annotations", "Annotations", "labels", "LABELS"):
            candidates.append((root / subdir / image.stem).with_suffix(ext))
    matches = [c for c in dict.fromkeys(candidates) if c.is_file()]
    if len(matches) == 1:
        return matches[0], False
    if len(matches) > 1:
        # If the same image has both TXT and XML, prefer XML because it
        # includes class names and avoids guessing YOLO class-index order.
        xmls = [c for c in matches if c.suffix == ".xml"]
        if len(xmls) == 1:
            return xmls[0], False
        return None, True

    matches = by_stem.get(image.stem, [])
    if len(matches) == 1:
        return matches[0], False
    split = split_for(relative)
    split_matches = [
        c for c in matches
        if split != "unspecified" and split_for(c.relative_to(root)) == split
    ]
    if len(split_matches) == 1:
        return split_matches[0], False
    return None, len(matches) > 1


def annotation_classes(path):
    """Count ground-truth objects; returns (Counter, error message)."""
    counts = collections.Counter()
    try:
        if path.suffix.lower() == ".xml":
            xml_root = ET.parse(path).getroot()
            for obj in xml_root.findall("object"):
                name = obj.findtext("name")
                box = obj.find("bndbox")
                if not name or box is None:
                    raise ValueError("VOC object missing name or bounding box")
                coords = [
                    float(box.findtext(tag))
                    for tag in ("xmin", "ymin", "xmax", "ymax")
                ]
                if coords[2] <= coords[0] or coords[3] <= coords[1]:
                    raise ValueError("VOC box width or height is not positive")
                counts[name.lower().strip().replace("-", "_").replace(" ", "_")] += 1
        else:
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"YOLO line {line_no} has {len(parts)} columns")
                class_id = int(parts[0])
                if class_id < 0:
                    raise ValueError(f"YOLO line {line_no} negative class ID")
                xywh = list(map(float, parts[1:]))
                if any(not 0 <= x <= 1 for x in xywh):
                    raise ValueError(f"YOLO line {line_no} normalized value outside 0..1")
                counts[f"class_id_{class_id}"] += 1
    except (ValueError, TypeError, ET.ParseError, OSError) as error:
        return collections.Counter(), str(error)
    return counts, None


def audit(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return {"root": str(root), "error": "Directory not found"}
    images = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".xml", ".txt"}
        and "imagesets" not in [s.lower() for s in p.parts]
    )
    by_stem = collections.defaultdict(list)
    for label in labels:
        by_stem[label.stem].append(label)
    split_images, split_pairs = collections.Counter(), collections.Counter()
    classes = collections.Counter()
    sample_pairs = []
    missing = []
    ambiguous = []
    bad_labels = []
    for image in images:
        split = split_for(image.relative_to(root))
        split_images[split] += 1
        annotation, conflict = paired_label(image, root, by_stem)
        if annotation is None:
            (ambiguous if conflict else missing).append(str(image.relative_to(root)))
            continue
        split_pairs[split] += 1
        if len(sample_pairs) < 6:
            sample_pairs.append({
                "image": str(image.relative_to(root)),
                "annotation": str(annotation.relative_to(root)),
                "split": split,
            })
        label_classes, error = annotation_classes(annotation)
        if error:
            bad_labels.append({
                "annotation": str(annotation.relative_to(root)),
                "error": error,
            })
        else:
            classes.update(label_classes)
    names_from_xml = {name for name in classes if not name.startswith("class_id_")}
    return {
        "root": str(root),
        "images": len(images),
        "annotation_files": len(labels),
        "matched_image_label_pairs": sum(split_pairs.values()),
        "image_splits": dict(sorted(split_images.items())),
        "paired_splits": dict(sorted(split_pairs.items())),
        "object_counts": dict(sorted(classes.items())),
        "all_six_classes_present_in_xml": EXPECTED.issubset(names_from_xml),
        "missing_annotations": len(missing),
        "ambiguous_annotations": len(ambiguous),
        "invalid_annotations": len(bad_labels),
        "sample_pairs": sample_pairs,
        "sample_missing": missing[:5],
        "sample_ambiguous": ambiguous[:5],
        "sample_invalid": bad_labels[:5],
        "dataset_yamls": [
            str(p.relative_to(root)) for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
        ][:10],
        "warning": (
            "A folder named test does not prove held-out independence. "
            "Confirm its images were never used to train or tune the NCNN model."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots", nargs="+", default=["datasets"],
        help="One or more existing dataset roots to inspect (read-only)",
    )
    parser.add_argument("--json", action="store_true",
                        help="Print machine-readable JSON output")
    args = parser.parse_args()
    results = [audit(root) for root in args.roots]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for entry in results:
            print(f"\n=== DATASET ROOT: {entry['root']} ===")
            if "error" in entry:
                print("ERROR:", entry["error"])
                continue
            for key in (
                "images", "annotation_files", "matched_image_label_pairs",
                "image_splits", "paired_splits", "object_counts",
                "missing_annotations", "ambiguous_annotations",
                "invalid_annotations", "dataset_yamls",
            ):
                print(f"{key}: {entry[key]}")
            for pair in entry["sample_pairs"]:
                print("EXAMPLE:", pair["image"], "->", pair["annotation"])
            for bad in entry["sample_invalid"]:
                print("INVALID:", bad)
            for name in entry["sample_missing"]:
                print("MISSING LABEL:", name)
            if entry["paired_splits"].get("test"):
                print("TEST-SPLIT CANDIDATES FOUND; verify original training split.")
            elif entry["matched_image_label_pairs"]:
                print("ANNOTATED IMAGES FOUND, but no explicit test split verified.")
            else:
                print("NO USABLE LABELED TEST IMAGES found under this directory.")
            print("CAUTION:", entry["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
