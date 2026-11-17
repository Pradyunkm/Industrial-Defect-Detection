"""Static dashboard checks that do not require a real camera or AI model load."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src/dashboard/app_pi.py"
CAMERAS = ROOT / "src/dashboard/camera_sources.py"
BROWSER = ROOT / "src/dashboard/browser_live.py"


class DashboardSourceTests(unittest.TestCase):
    def test_dashboard_python_syntax_compiles(self):
        source = APP.read_text(encoding="utf-8")
        compile(source, str(APP), "exec")
        compile(CAMERAS.read_text(encoding="utf-8"), str(CAMERAS), "exec")
        compile(BROWSER.read_text(encoding="utf-8"), str(BROWSER), "exec")

    def test_upload_live_camera_and_history_controls_exist(self):
        source = APP.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertTrue(any(isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "tabs"
                            for node in ast.walk(tree)))
        for feature in (
            "st.file_uploader(", "st.camera_input(", "open_camera(",
            "Show YOLO bounding-box annotations",
            "Start live detection", "Stop and review latest frame",
            "Download live-frame annotations", "Inspection history",
            "USB webcam on Raspberry Pi", "Raspberry Pi Camera Module (CSI)",
            "RTSP / HTTP network camera", "Browser webcam (Windows or phone)",
        ):
            with self.subTest(feature=feature):
                self.assertIn(feature, source)

    def test_optional_camera_modules_contain_expected_backends(self):
        camera_code = CAMERAS.read_text(encoding="utf-8")
        browser_code = BROWSER.read_text(encoding="utf-8")
        for feature in ("class OpenCVCamera", "class Picamera2Camera",
                        "class RpicamStillCamera", "def open_camera",
                        "rpicam-still", "camera_inventory"):
            with self.subTest(feature=feature):
                self.assertIn(feature, camera_code)
        for feature in ("streamlit_webrtc", "VideoProcessorBase",
                        "browser webcam frame", "webrtc_streamer"):
            with self.subTest(feature=feature):
                self.assertIn(feature.lower(), browser_code.lower())


if __name__ == "__main__":
    unittest.main()
