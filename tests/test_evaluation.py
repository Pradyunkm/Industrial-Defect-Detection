"""Test held-out evaluation math and label readers without loading model weights."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_dataset import (
    average_precision, canonical, iou_xyxy, label_path_for,
    metrics_for_class, parse_voc, parse_yolo,
)


class DetectionEvaluationTests(unittest.TestCase):
    def test_iou_perfect_overlap(self):
        self.assertAlmostEqual(iou_xyxy([0, 0, 20, 20], [0, 0, 20, 20]), 1)

    def test_iou_disjoint_boxes(self):
        self.assertEqual(iou_xyxy([0, 0, 2, 2], [3, 3, 5, 5]), 0)

    def test_class_normalization(self):
        self.assertEqual(canonical("Rolled-in Scale"), "rolled_in_scale")

    def test_ap_all_true_positives(self):
        self.assertAlmostEqual(average_precision([True, True], 2), 1)

    def test_ap_no_true_positives(self):
        self.assertEqual(average_precision([False], 2), 0)

    def test_ap_no_ground_truth_is_undefined(self):
        self.assertIsNone(average_precision([False], 0))

    def test_class_metrics_counts_fp_and_fn(self):
        gt = {
            "img.jpg": [
                {"class": "scratches", "box": [0, 0, 10, 10]},
                {"class": "scratches", "box": [30, 30, 40, 40]},
            ]
        }
        pred = [
            {"image": "img.jpg", "class": "scratches", "confidence": 0.9,
             "box": [0, 0, 10, 10]},
            {"image": "img.jpg", "class": "scratches", "confidence": 0.8,
             "box": [0, 0, 10, 10]},
        ]
        result = metrics_for_class(pred, gt, "scratches")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)

    def test_pascal_voc_xml_parser(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s.xml"
            path.write_text(
                "<annotation><object><name>scratches</name><bndbox>"
                "<xmin>1</xmin><ymin>2</ymin><xmax>15</xmax>"
                "<ymax>20</ymax></bndbox></object></annotation>",
                encoding="utf-8",
            )
            self.assertEqual(parse_voc(path)[0]["box"], [1, 2, 15, 20])

    def test_yolo_normalized_parser(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s.txt"
            path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            annotations = parse_yolo(path, 100, 100, {0: "scratches"})
            self.assertAlmostEqual(annotations[0]["box"][0], 40)
            self.assertAlmostEqual(annotations[0]["box"][2], 60)

    def test_unlabeled_images_raise_instead_of_being_called_normal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            img = images / "image.jpg"
            img.write_bytes(b"test")
            with self.assertRaises(FileNotFoundError):
                label_path_for(img, images, labels)


if __name__ == "__main__":
    unittest.main()
