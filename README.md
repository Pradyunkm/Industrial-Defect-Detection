# Industrial Defect Detection & Quality-Control Assistant

**Bharat AI-SoC Challenge 2026–27 · Problem Statement 5**

An **offline-capable steel-surface inspection research prototype** running on a Raspberry Pi 5. It combines YOLOv8 Nano (exported to NCNN) for image-based defect detection with Llama 3.2 1B Instruct (via ExecuTorch) for optional, on-device inspection advice.

> **Research prototype, not a certified quality-control system.** Detection confidence is not defect severity; no detections do not prove a defect-free surface. Human verification is required.

## System architecture

```text
Uploaded steel image OR Pi-connected USB / network camera frame
    |
    v
YOLOv8 Nano / NCNN (Raspberry Pi CPU)
    |
    +--> Defect label, bounding box, confidence
    |           |
    |           +--> Demonstration-only review rules
    |           |
    |           +--> Optional ExecuTorch Llama 3.2 1B
    |                        |
    |                        +--> Advisory inspection explanation
    |                              (rule-based fallback if unsuitable)
    v
Streamlit dashboard
    +--> Original and annotated image
    +--> Annotated PNG download
    +--> Save inspection to local SQLite
    +--> Inspection history and CSV export
```

Llama receives a **text summary of YOLO detections**, not the image pixels. It is invoked on demand, not on every camera frame. The Raspberry Pi dashboard supports **image uploads**, optional **Pi-side USB/RTSP live video inspection**, and browser-camera **still-photo capture**. The separate Windows development application's camera code is unchanged and not included here. **Pi live-camera operation has not yet been hardware-verified.**

## Detected defect classes

The model uses the six NEU-DET categories:

1. Crazing
2. Inclusion
3. Patches
4. Pitted surface
5. Rolled-in scale
6. Scratches

Predictions are model outputs, not verified physical measurements. The demonstration review labels (such as `HIGH REVIEW`) are heuristic prompts for human review, **not calibrated severity scores or pass/fail decisions**.

## Hardware and environment

This deployment was developed on:

- Raspberry Pi 5 with 8 GB RAM, ARM64 Raspberry Pi OS / Debian
- Python 3.13 virtual environment for the YOLO/Streamlit application
- Ultralytics YOLOv8 Nano exported to NCNN
- ExecuTorch 1.5.0, built separately for ARM64, with a Llama 3.2 1B Instruct `.pte` model

See [requirements.txt](requirements.txt) for observed Python package versions. PyTorch on the Pi uses a specialized ARM64 installation; do **not** assume the generic PyPI wheel exactly reproduces it. ExecuTorch is a **separate environment and native build**, not installed by this project's `requirements.txt` alone.

A stable power source is important for repeatable inference. Development with laptop USB-C power previously produced undervoltage warnings; a suitable dedicated Pi 5 supply is recommended for demonstrations.

## Repository layout

```text
src/
  dashboard/app_pi.py       Streamlit upload, multiple live feeds and inspection history
  dashboard/camera_sources.py  USB/V4L2, Pi CSI, and RTSP/HTTP backends
  dashboard/browser_live.py  Optional browser webcam over WebRTC
  inference/common.py       YOLO inference and demo review rules
  inference/infer_image.py  Single-image command-line inference
  inference/llama_explain.py  Optional offline Llama explanation helper
  inference/inspection_store.py  Shared SQLite, PNG and CSV persistence
scripts/evaluate_dataset.py  Labeled held-out test-set evaluation
tests/                     Logic, camera, persistence and evaluation tests
docs/TECHNICAL_REPORT.md   Report draft with confirmed and pending results
docs/DEMO_SCRIPT.md        Narrated demonstration recording plan
docs/SUBMISSION_CHECKLIST.md  Final submission and verification checklist
README.md
requirements.txt
requirements-camera.txt    Optional browser WebRTC dependencies
tests/test_offline_logic.py  Lightweight offline logic tests
.gitignore
```

Large model files, training datasets, run outputs, local databases, virtual environments, and generated inspection images are deliberately excluded from Git.

## Required local model paths

The application expects the exported **NCNN model directory** at:

```text
~/Industrial_Defect_Detection/models/exported/best_ncnn_model/
```

The Llama helper expects these separate, local components:

```text
~/executorch-deployment/executorch/cmake-out/examples/models/llama/llama_main
~/executorch-deployment/models/llama3.2-1b/Llama-3.2-1B-Instruct.pte
~/executorch-deployment/models/llama3.2-1b/tokenizer.model
```

The compatible Llama artifacts used in the prototype came from [executorch-community/Llama-3.2-1B-Instruct-ET](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-ET). Review applicable model licenses and access terms before redistributing models. **Cloning this repository alone is not sufficient to run inference**; the models, dependencies, and ExecuTorch native runner must be provisioned separately.

## Run the Raspberry Pi dashboard

After setting up the Python virtual environment, dependencies, NCNN model, and ExecuTorch runner:

```bash
cd ~/Industrial_Defect_Detection
source .venv/bin/activate
python -m streamlit run src/dashboard/app_pi.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

From a computer on the same local network, open `http://<raspberry-pi-ip>:8501`.

In the **Upload image** tab, upload a steel-surface JPG or PNG (or capture a **single still photo** using the browser camera), review YOLO detections, optionally choose **Explain with Llama (offline)**, then download the annotated PNG or save the inspection to SQLite. The **Inspection history** tab displays saved records and supports CSV export.

### Experimental live camera with automatic annotations

Open the **Live camera** tab and choose one of four sources:

1. **USB webcam on Raspberry Pi** — V4L2 / OpenCV, choose index 0, 1, etc.; useful for a USB webcam or HDMI USB capture dongle connected to the Pi
2. **Raspberry Pi Camera Module (CSI)** — official or compatible ribbon-cable camera on the Pi 5 CAM0/CAM1 connector; uses `Picamera2` if importable from the app Python, otherwise tries the system `rpicam-still` capture command (slower, periodic still images)
3. **RTSP / HTTP network camera** — an IP camera with an OpenCV-readable stream URL
4. **Browser webcam (Windows or phone) — WebRTC** — optional `streamlit-webrtc`/PyAV installation; reads the camera of the device displaying the dashboard and runs inference on the Pi

For Pi-side USB, CSI or network cameras, click **Start live detection**. If the selected source is available, the page periodically shows camera frames with YOLO bounding boxes, defect labels, confidence scores and a demonstration review category. Use **Show YOLO bounding-box annotations** in the sidebar to toggle the overlay. The **Detect attached Pi cameras** button lists Linux V4L2 devices and uses `rpicam-hello --list-cameras` when available.

Click **Stop and review latest frame** to freeze the last successfully analyzed frame. You can then download its annotated PNG, request optional offline Llama guidance, or save the snapshot and detection metadata into SQLite.

**Camera placement and browser security:** USB, CSI and IP video sources are opened by the **Raspberry Pi server**. A Windows laptop's built-in camera does not appear as a Pi USB camera. To use it, choose **Browser webcam — WebRTC**; this requires the optional `streamlit-webrtc` package (and its PyAV dependency) in the Pi app environment:

```bash
# Optional only; do not install if you only need Pi-side USB/CSI/IP cameras
python -m pip install streamlit-webrtc
```

Browser media permissions require a **secure context** (HTTPS or `localhost`). For local Windows development, use an SSH local port-forward in Windows PowerShell:

```powershell
ssh -L 8502:127.0.0.1:8501 pi@pi.local
```

Keep the SSH tunnel open, and visit `http://localhost:8502` **in the Windows browser**. This can enable camera permission on the laptop without exposing the Pi app to the public internet. WebRTC's peer connection may still require network setup; this prototype requests local ICE candidates only and does not use a public STUN/TURN server. Alternatively, the **Upload image** tab's browser still-photo option supports a single frame.

Camera Module drivers and `rpicam-*` tools are **system packages**; consult the [official Raspberry Pi camera software guide](https://www.raspberrypi.com/documentation/accessories/camera-software.html) for compatible installation. On systems where Picamera2 is installed only into the system Python, the app's virtual environment might not import it; the `rpicam-still` fallback can work without modifying the virtual environment.

**Performance/testing:** Continuous YOLO annotation is CPU-limited and updates periodically, not at 30 FPS. The optional browser webcam also throttles inference to reduce load. Use a stable dedicated Pi 5 supply before extended testing. **No USB, CSI, IP or browser live camera hardware has been verified yet in this project; the added paths are experimental.**

The local database is `database/inspections.db`; annotated saved outputs are under `logs/annotated/`. These are runtime artifacts and are not committed.

### Single-image command line

```bash
cd ~/Industrial_Defect_Detection
source .venv/bin/activate
python src/inference/infer_image.py \
  --model models/exported/best_ncnn_model \
  --image /path/to/steel_image.jpg \
  --output logs/annotated/pi_output.png
```

The command writes an annotated PNG and a JSON report containing detected regions and demonstration review information.

## What has been demonstrated

- YOLO NCNN inference on the Raspberry Pi CPU, including annotated images and confidence scores
- ExecuTorch Llama 3.2 1B text generation on the Raspberry Pi CPU
- **Combined network-isolated YOLO → Llama inference** on the Raspberry Pi CPU using the `scratches_271.jpg` image; three `scratches` regions detected with confidence scores 0.641, 0.568 and 0.432, followed by an ExecuTorch Llama inspection recommendation
- The combined test reported **1,416 ms YOLO inference** for that one image (not a performance benchmark)
- Network isolation separately verified under the same `sudo unshare --net -- runuser -u pi` setup: TCP connection to `1.1.1.1:443` failed with `[Errno 101] Network is unreachable`
- **8/8 lightweight automated tests passing**, covering YOLO review rules and mocked Llama response handling
- Combined YOLO → Llama explanations inside the Streamlit dashboard
- SQLite inspection saving, annotated-file persistence, and CSV generation/download
- Multi-source live-camera UI code added for USB, CSI, IP and optional browser WebRTC; **hardware test pending**

**Still to verify:** the new multi-source live-camera UI with real USB, CSI, IP or browser camera hardware, the full dashboard workflow under network isolation, systematic evaluation across the six defect classes, and repeatable performance under a stable dedicated power supply. No overall accuracy or real-time camera-performance claim is made here.

### Run the lightweight logic tests

```bash
cd ~/Industrial_Defect_Detection
.venv/bin/python -m unittest discover -s tests -v
```

These tests do not load model weights, access the internet, or require an attached camera.

## Evaluate a held-out labeled test set

To calculate **per-class precision, recall, F1 and AP50**, put actual *held-out* NEU-DET images and their matching Pascal VOC XML or YOLO TXT labels into separate directories. Replace the example paths with your real folder names:

```bash
cd ~/Industrial_Defect_Detection
source .venv/bin/activate
python scripts/evaluate_dataset.py \
  --images datasets/processed/test/images \
  --labels datasets/processed/test/labels \
  --model models/exported/best_ncnn_model \
  --output logs/evaluation
```

The evaluator checks that each image has a matching annotation; it refuses to treat missing labels as normal images. Results are saved as `logs/evaluation/metrics.json` and `logs/evaluation/per_class_metrics.csv`. It reports **AP50 only**, not mAP50-95. Six-class mAP50 requires test examples for all six classes.

**Do not report accuracy until the actual test-set evaluation has run**, and avoid lengthy evaluation on laptop USB-C power.

## Final documentation

- [Technical report draft](docs/TECHNICAL_REPORT.md)
- [Demonstration video script](docs/DEMO_SCRIPT.md)
- [Submission and browser verification checklist](docs/SUBMISSION_CHECKLIST.md)
- Optional WebRTC package versions: [requirements-camera.txt](requirements-camera.txt)

Newer versions of the code also save Llama explanation text alongside annotated upload and camera snapshots. Run the updated regression suite and confirm the behavior in the browser before treating these changes as verified.

## Safety and limitations

- The NEU-DET training/evaluation setup does not establish reliable acceptance of normal (defect-free) surfaces.
- YOLO bounding boxes and confidences are **not** real-world dimensions, defect causes, or calibrated severity.
- Llama can generate incomplete or unsupported advice. The prototype uses a limited output filter and a class-specific rule-based fallback, but neither guarantees factual correctness.
- Human inspection and approved site-specific criteria must determine any production QC disposition.
- The Pi dashboard now has an **experimental** live-camera tab for Pi-connected USB, CSI and IP sources plus optional browser WebRTC. Live-camera hardware operation has not yet been tested, and any separate Windows camera implementation must be evaluated independently.

## Next work

Verify the **new live-camera tab with compatible camera hardware**, expand quantitative defect-class evaluation, test the complete dashboard/browser workflow, and prepare architecture screenshots and a demonstration video for the challenge submission.
