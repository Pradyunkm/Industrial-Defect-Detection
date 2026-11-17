"""Optional camera inputs for the Raspberry Pi Streamlit dashboard.

All camera devices are opened only after a user clicks Start.
Outputs are OpenCV BGR numpy frames for the existing YOLO pipeline.
"""
from pathlib import Path
from shutil import which
import subprocess

import cv2
import numpy as np


class CameraOpenError(RuntimeError):
    """Camera could not be opened or a required backend is unavailable."""


class OpenCVCamera:
    def __init__(self, source, usb=False):
        self.capture = None
        if usb:
            self.capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(source)
        else:
            self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise CameraOpenError(f"OpenCV cannot open camera source: {source}")
        if usb:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def isOpened(self):
        return self.capture is not None and self.capture.isOpened()

    def read(self):
        return self.capture.read()

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class Picamera2Camera:
    """Pi CSI camera using optional system libcamera/Picamera2 bindings."""

    def __init__(self, index=0):
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise CameraOpenError("Picamera2 is not importable in this virtual environment") from error

        self.camera = None
        try:
            self.camera = Picamera2(camera_num=index)
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
        except Exception as error:
            self.release()
            raise CameraOpenError(f"Picamera2 failed to open CSI camera {index}: {error}") from error

    def isOpened(self):
        return self.camera is not None

    def read(self):
        if self.camera is None:
            return False, None
        try:
            # Picamera2 RGB888 array bytes are BGR, directly usable by OpenCV.
            frame_bgr = self.camera.capture_array()
            return frame_bgr is not None, frame_bgr
        except Exception:
            return False, None

    def release(self):
        if self.camera is not None:
            try:
                self.camera.stop()
            finally:
                self.camera.close()
                self.camera = None


class RpicamStillCamera:
    """CSI fallback: repeatedly capture stills with the system rpicam-still CLI.

    This is a slow preview fallback, not a video-rate capture interface.
    """

    def __init__(self, index=0):
        self.command = which("rpicam-still")
        if not self.command:
            raise CameraOpenError(
                "Neither importable Picamera2 nor rpicam-still was found. "
                "Install/configure Raspberry Pi camera software separately."
            )
        self.index = int(index)
        self.active = True

    def isOpened(self):
        return self.active

    def read(self):
        if not self.active:
            return False, None
        try:
            result = subprocess.run(
                [self.command, "--nopreview", "--timeout", "300",
                 "--camera", str(self.index),
                 "--width", "640", "--height", "480",
                 "--encoding", "jpg", "--output", "-"],
                capture_output=True, timeout=12, check=False,
            )
            if result.returncode != 0:
                return False, None
            frame_bgr = cv2.imdecode(
                np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            return frame_bgr is not None, frame_bgr
        except (subprocess.TimeoutExpired, OSError):
            return False, None

    def release(self):
        self.active = False


def open_camera(kind, *, index=0, url=""):
    """Open USB/V4L2, CSI or RTSP/HTTP camera; release via camera.release()."""
    if kind == "USB webcam on Raspberry Pi":
        return OpenCVCamera(index, usb=True)

    if kind == "Raspberry Pi Camera Module (CSI)":
        try:
            return Picamera2Camera(index)
        except CameraOpenError as picamera_error:
            try:
                return RpicamStillCamera(index)
            except CameraOpenError as cli_error:
                raise CameraOpenError(
                    f"CSI camera unavailable: {picamera_error}; {cli_error}"
                ) from cli_error

    if kind == "RTSP / HTTP network camera":
        if not url.startswith(("rtsp://", "http://", "https://")):
            raise CameraOpenError("Network camera URL must start with rtsp://, http:// or https://")
        return OpenCVCamera(url, usb=False)

    raise CameraOpenError(f"Unsupported camera type: {kind}")


def camera_inventory():
    """Return a short text listing of local camera device nodes and CSI tools."""
    video_nodes = sorted(str(p) for p in Path("/dev").glob("video[0-9]*"))
    details = ["USB/V4L2 devices: " + (", ".join(video_nodes) if video_nodes else "none found")]
    tool = which("rpicam-hello")
    if tool:
        try:
            result = subprocess.run(
                [tool, "--list-cameras"], capture_output=True, text=True,
                timeout=8, check=False
            )
            details.append("CSI camera inventory:\n" + (result.stdout + result.stderr).strip()[-3000:])
        except (OSError, subprocess.TimeoutExpired) as error:
            details.append(f"CSI inventory unavailable: {error}")
    else:
        details.append("CSI discovery: rpicam-hello not installed")
    return "\n".join(details)
