"""Read-only NEU-DET audit tests; no model inference, network, or real datasets."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_dataset import audit, annotation_classes, paired_label, split_for


XML = """<annotation><object><name>rolled-in_scale</name><bndbox>
<xmin>2</xmin><ymin>3</ymin><xmax>30</xmax><ymax>40</ymax>
</bndbox></object></annotation>"""


class DatasetAuditTests(unittest.TestCase):
    def test_detect_explicit_test_split(self):
        self.assertEqual(split_for(Path("a/test/images/im.jpg")), "test")
        self.assertEqual(split_for(Path("a/train/images/im.jpg")), "train")
        self.assertEqual(split_for(Path("a/IMAGES/im.jpg")), "unspecified")

    def test_flat_neu_xml_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_folder = root / "IMAGES"
            label_folder = root / "ANNOTATIONS"
            image_folder.mkdir()
            label_folder.mkdir()
            image = image_folder / "rolled-in_scale_1.jpg"
            image.write_bytes(b"image-fixture")
            xml = label_folder / "rolled-in_scale_1.xml"
            xml.write_text(XML, encoding="utf-8")
            result = audit(root)
            self.assertEqual(result["matched_image_label_pairs"], 1)
            self.assertEqual(result["missing_annotations"], 0)
            self.assertEqual(result["object_counts"]["rolled_in_scale"], 1)
            self.assertEqual(result["paired_splits"]["unspecified"], 1)

    def test_split_yolo_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_folder = root / "test/images"
            label_folder = root / "test/labels"
            image_folder.mkdir(parents=True)
            label_folder.mkdir(parents=True)
            image = image_folder / "one.jpg"
            image.write_bytes(b"test-fixture")
            (label_folder / "one.txt").write_text(
                "5 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            result = audit(root)
            self.assertEqual(result["paired_splits"]["test"], 1)
            self.assertEqual(result["object_counts"]["class_id_5"], 1)

    def test_unpaired_image_reported_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "nothing.jpg").write_bytes(b"image")
            result = audit(root)
            self.assertEqual(result["images"], 1)
            self.assertEqual(result["missing_annotations"], 1)

    def test_invalid_xml_box_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            xml = Path(folder) / "bad.xml"
            xml.write_text(XML.replace("<xmax>30</xmax>",
                                       "<xmax>1</xmax>"), encoding="utf-8")
            counts, error = annotation_classes(xml)
            self.assertFalse(counts)
            self.assertIn("positive", error)

    def test_ambiguous_duplicate_label_filenames_not_guessed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_folder = root / "IMAGES"
            image_folder.mkdir()
            image = image_folder / "one.jpg"
            image.write_bytes(b"test")
            (root / "a").mkdir()
            (root / "b").mkdir()
            for parent in ("a", "b"):
                (root / parent / "one.xml").write_text(XML, encoding="utf-8")
            result = audit(root)
            self.assertEqual(result["ambiguous_annotations"], 1)
            self.assertEqual(result["matched_image_label_pairs"], 0)

    def test_missing_dataset_path_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            result = audit(Path(folder) / "does-not-exist")
            self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
