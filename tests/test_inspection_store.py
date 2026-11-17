"""Test SQLite and CSV persistence without running YOLO, Llama, or a camera."""

import csv
import io
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/inference"))
from inspection_store import (
    export_inspections_csv, init_db, load_inspections, save_inspection,
)


class InspectionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.db = root / "database/inspections.db"
        self.output = root / "logs/annotated"

    def test_initializing_database_is_repeatable(self):
        init_db(self.db)
        init_db(self.db)
        self.assertEqual(load_inspections(self.db), [])

    def test_image_and_llama_guidance_save_to_database(self):
        inspection_id = save_inspection(
            self.db, self.output, filename="scratches_271.jpg",
            model="models/exported/best_ncnn_model", conf=0.3,
            imgsz=320,
            detections=[{"class": "scratches", "confidence": 0.64}],
            priority="MEDIUM REVIEW",
            recommendation="Inspect the scratches for discoloration.",
            annotated_png=b"simulated-png-test-bytes",
        )
        self.assertTrue((self.output / (inspection_id + ".png")).is_file())
        records = load_inspections(self.db)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["recommendation"], "Inspect the scratches for discoloration.")
        self.assertEqual(records[0]["detection_count"], 1)
        self.assertEqual(records[0]["filename"], "scratches_271.jpg")

    def test_filename_is_normalized_to_basename(self):
        save_inspection(
            self.db, self.output, filename="../../untrusted.jpg",
            model="ncnn", conf=0.3, imgsz=320,
            detections=[], priority="UNVERIFIED",
            recommendation="Further inspection needed.",
            annotated_png=b"example",
        )
        self.assertEqual(load_inspections(self.db)[0]["filename"], "untrusted.jpg")

    def test_csv_header_and_saved_rows(self):
        save_inspection(
            self.db, self.output, filename="example.png",
            model="ncnn", conf=0.3, imgsz=320,
            detections=[], priority="UNVERIFIED",
            recommendation="Human inspection required.",
            annotated_png=b"example",
        )
        text = export_inspections_csv(load_inspections(self.db))
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["filename"], "example.png")
        self.assertEqual(rows[0]["recommendation"], "Human inspection required.")
        self.assertIn("timestamp_utc", rows[0])

    def test_empty_image_cannot_be_saved(self):
        with self.assertRaises(ValueError):
            save_inspection(
                self.db, self.output, filename="bad.png",
                model="ncnn", conf=0.3, imgsz=320,
                detections=[], priority="UNVERIFIED",
                recommendation="Inspect", annotated_png=b"",
            )
        self.assertEqual(load_inspections(self.db), [])


if __name__ == "__main__":
    unittest.main()
