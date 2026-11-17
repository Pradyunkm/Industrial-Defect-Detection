"""Lightweight offline checks: no model inference or network required."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "inference"))

from common import review
from llama_explain import explain_defects


class ReviewRuleTests(unittest.TestCase):
    def test_no_detections_are_unverified(self):
        status, message = review([])
        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("not proof", message)

    def test_crazing_triggers_high_review(self):
        status, _ = review([{"class": "crazing", "bounding_box_fraction": 0.01}])
        self.assertEqual(status, "HIGH REVIEW")

    def test_large_bounding_box_triggers_high_review(self):
        status, _ = review([{"class": "scratches", "bounding_box_fraction": 0.15}])
        self.assertEqual(status, "HIGH REVIEW")

    def test_medium_bounding_box_triggers_medium_review(self):
        status, _ = review([{"class": "patches", "bounding_box_fraction": 0.06}])
        self.assertEqual(status, "MEDIUM REVIEW")

    def test_small_bounding_box_requires_review(self):
        status, _ = review([{"class": "scratches", "bounding_box_fraction": 0.01}])
        self.assertEqual(status, "REVIEW")


class LlamaHelperTests(unittest.TestCase):
    @staticmethod
    def fake_process(text):
        def respond(command, **kwargs):
            assert kwargs.get("timeout") == 180
            prompt = command[command.index("--prompt") + 1]
            return SimpleNamespace(
                stdout=prompt + text + "<|eot_id|>\nPyTorchObserver {}"
            )
        return respond

    @patch("llama_explain.subprocess.run")
    def test_prompt_uses_yolo_labels_and_removes_special_tokens(self, run):
        run.side_effect = self.fake_process(
            "Inspect the marked scratches for linear surface marks."
        )
        report = explain_defects([
            {"class": "scratches", "confidence": 0.64},
            {"class": "scratches", "confidence": 0.57},
        ])
        self.assertIn("2 possible scratches", report["summary"])
        self.assertEqual(
            report["llama_explanation"],
            "Inspect the marked scratches for linear surface marks.",
        )
        self.assertNotIn("<|eot_id|>", report["llama_explanation"])

    @patch("llama_explain.subprocess.run")
    def test_unsupported_measurement_uses_fixed_fallback(self, run):
        run.side_effect = self.fake_process("This scratch is 2 cm long.")
        report = explain_defects([{"class": "scratches"}])
        self.assertIn("Visually inspect the marked regions", report["llama_explanation"])
        self.assertNotIn("2 cm", report["llama_explanation"])

    @patch("llama_explain.subprocess.run")
    def test_empty_model_response_uses_fixed_fallback(self, run):
        run.side_effect = self.fake_process("")
        report = explain_defects([{"class": "crazing"}])
        self.assertIn("Visually inspect the marked regions", report["llama_explanation"])


if __name__ == "__main__":
    unittest.main()
