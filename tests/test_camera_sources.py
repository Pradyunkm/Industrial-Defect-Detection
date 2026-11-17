"""Mock-based camera selector tests; no hardware or video feed is opened."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/dashboard"))

from camera_sources import CameraOpenError, OpenCVCamera, open_camera


class CameraSelectionTests(unittest.TestCase):
    @patch("camera_sources.cv2.VideoCapture")
    def test_usb_webcam_uses_local_device_index(self, capture_factory):
        fake = MagicMock()
        fake.isOpened.return_value = True
        capture_factory.return_value = fake
        camera = open_camera("USB webcam on Raspberry Pi", index=0)
        self.assertIsInstance(camera, OpenCVCamera)
        self.assertTrue(camera.isOpened())
        self.assertEqual(capture_factory.call_args.args[0], 0)
        camera.release()
        fake.release.assert_called()

    @patch("camera_sources.Picamera2Camera")
    def test_csi_prefers_picamera2(self, picamera_factory):
        instance = open_camera("Raspberry Pi Camera Module (CSI)", index=1)
        picamera_factory.assert_called_once_with(1)
        self.assertIs(instance, picamera_factory.return_value)

    @patch("camera_sources.RpicamStillCamera")
    @patch("camera_sources.Picamera2Camera")
    def test_csi_falls_back_to_rpicam_cli(self, picamera_factory, still_factory):
        picamera_factory.side_effect = CameraOpenError("No Picamera2")
        camera = open_camera("Raspberry Pi Camera Module (CSI)", index=0)
        still_factory.assert_called_once_with(0)
        self.assertIs(camera, still_factory.return_value)

    @patch("camera_sources.cv2.VideoCapture")
    def test_ip_camera_uses_open_cv_stream(self, capture_factory):
        fake = MagicMock()
        fake.isOpened.return_value = True
        capture_factory.return_value = fake
        url = "rtsp://192.168.1.2:554/stream"
        camera = open_camera("RTSP / HTTP network camera", url=url)
        self.assertIsInstance(camera, OpenCVCamera)
        capture_factory.assert_called_once_with(url)
        camera.release()

    def test_invalid_ip_scheme_is_rejected(self):
        with self.assertRaises(CameraOpenError):
            open_camera("RTSP / HTTP network camera", url="file:///etc/passwd")

    def test_unknown_camera_source_is_rejected(self):
        with self.assertRaises(CameraOpenError):
            open_camera("Unsupported video source")


if __name__ == "__main__":
    unittest.main()
