"""Static dashboard checks that do not require a real camera or AI model load."""

import ast
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "src/dashboard/app_pi.py"


class DashboardSourceTests(unittest.TestCase):
    def test_dashboard_python_syntax_compiles(self):
        source = APP.read_text(encoding="utf-8")
        compile(source, str(APP), "exec")

    def test_upload_live_camera_and_history_controls_exist(self):
        source = APP.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertTrue(any(isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "tabs"
                            for node in ast.walk(tree)))
        for feature in (
            "st.file_uploader(", "st.camera_input(", "cv2.VideoCapture(",
            "Show YOLO bounding-box annotations",
            "Start live detection", "Stop and review latest frame",
            "Download live-frame annotations", "Inspection history",
        ):
            with self.subTest(feature=feature):
                self.assertIn(feature, source)


if __name__ == "__main__":
    unittest.main()
