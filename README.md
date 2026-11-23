# Industrial Defect Detection & Quality-Control Assistant

**Bharat AI-SoC Challenge 2026–27 · Problem Statement 5**

An **offline-capable steel-surface inspection research prototype** running on a Raspberry Pi 5. It combines an improved 100-epoch YOLO detector (exported to NCNN at 320 × 320) for image-based defect detection with Llama 3.2 1B Instruct (via ExecuTorch) for optional, on-device inspection advice.

> **Research prototype, not a certified quality-control system.** Detection confidence is not defect severity; no detections do not prove a defect-free surface. Human verification is required.

## System architecture

```text
Uploaded steel image OR Pi-connected USB / network camera frame
    |
    v
Improved 100-epoch YOLO / NCNN (Raspberry Pi CPU)
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
- Ultralytics 100-epoch YOLO detector (11.13 million parameters), exported to NCNN at 320 × 320; the original YOLOv8 Nano is retained for rollback
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
scripts/audit_dataset.py  Read-only inventory of annotations and split evidence
tests/                     Logic, camera, persistence and evaluation tests
docs/DATASET_EVALUATION.md   Six-class labeled-test preparation and provenance guide
docs/TECHNICAL_REPORT.md   Technical report with final measured results and limitations
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
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

For the verified Windows setup, create an SSH tunnel in a **separate Windows PowerShell terminal** with `ssh -N -L 8502:127.0.0.1:8501 pi@pi.local`, then open `http://127.0.0.1:8502` in the Windows browser. Keep both the Pi Streamlit process and Windows tunnel running.

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
ssh -N -L 8502:127.0.0.1:8501 pi@pi.local
```

Keep the SSH tunnel open, and visit `http://localhost:8502` **in the Windows browser**. This can enable camera permission on the laptop without exposing the Pi app to the public internet. WebRTC's peer connection may still require network setup; this prototype requests local ICE candidates only and does not use a public STUN/TURN server. Alternatively, the **Upload image** tab's browser still-photo option supports a single frame.

Camera Module drivers and `rpicam-*` tools are **system packages**; consult the [official Raspberry Pi camera software guide](https://www.raspberrypi.com/documentation/accessories/camera-software.html) for compatible installation. On systems where Picamera2 is installed only into the system Python, the app's virtual environment might not import it; the `rpicam-still` fallback can work without modifying the virtual environment.

**Performance/testing:** Continuous YOLO annotation is CPU-limited and updates periodically, not at 30 FPS. The optional browser webcam also throttles inference to reduce load. Use a stable dedicated Pi 5 supply before extended testing. **Windows browser WebRTC camera was verified by the user** in a successful manual website walkthrough; Pi-connected USB, CSI and RTSP/HTTP camera hardware remains untested. This is not a measured frame-rate or accuracy benchmark.

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
- **42/42 automated regression tests** passed on the Pi after deployment, covering camera-source logic, dataset audits, evaluation logic, SQLite persistence, and Llama handling. Earlier comprehensive offline verification passed **18/18 checks**; the latter included real network-isolated YOLO → Llama, annotated PNG, temporary SQLite and CSV verification
- **8/8 browser walkthrough checks passed**, including image upload, webcam feed, saving and history, as reported by the user
- Combined YOLO → Llama explanations inside the Streamlit dashboard
- SQLite inspection saving, annotated-file persistence, and CSV generation/download
- Multi-source live-camera UI code added for USB, CSI, IP and optional browser WebRTC; **browser WebRTC manual test passed**, Pi-side hardware testing pending

**Still to verify:** real Pi-connected USB/CSI/IP camera hardware, the full dashboard/browser workflow under network isolation, and sustained production-camera throughput on independent industrial data. Six-class NEU-DET test metrics are reported below; no real-time camera FPS or certified production acceptance is claimed.

## Latest deployed model and final evaluation (23 November 2026)

The deployed NCNN model is the improved **100-epoch YOLO checkpoint**, exported at **320 × 320** and installed at `models/exported/best_ncnn_model`. The previous YOLOv8 Nano NCNN model is preserved in `models/backups/yolov8n_original_ncnn` for rollback. These local model folders are not committed to GitHub.

Both models were evaluated using the same project evaluator on the **180-image NEU-DET test split**, with **446 annotated defect regions**, IoU 0.50, and operating confidence 0.30. The split corresponds to the documented 1,440-train / 180-validation / 180-test setup; filenames and byte-identical file checks found no train/validation-versus-test duplicates. The Pi evaluator did not independently verify provenance with `--train-images`; these checks do not exclude every possible form of leakage.

| Test-set measure | Original YOLOv8 Nano | Improved 100-epoch YOLO |
|---|---:|---:|
| mAP50 | 74.32% | **74.66%** |
| True positives at confidence 0.30 | 268 | **286** |
| False positives | 73 | **67** |
| False negatives | 178 | **160** |
| Micro F1 | 68.11% | **71.59%** |
| Raspberry Pi average prediction time | **76.5 ms** | 92.2 ms |

The latency comparison used 12 sequential predictions per model and **does not constitute a live-camera FPS measurement**. The improved model had better F1 at the selected threshold, but its gains in overall mAP50 were modest and it was slower.

| Defect | Original AP50 | Improved AP50 |
|---|---:|---:|
| Crazing | 44.61% | 46.03% |
| Inclusion | 79.55% | 81.50% |
| Patches | 92.16% | 92.78% |
| Pitted surface | 73.11% | 81.61% |
| Rolled-in scale | 67.78% | 64.02% |
| Scratches | 88.71% | 82.00% |

**Key limitation:** At confidence 0.30, only 17 of 79 annotated crazing regions were detected (**21.52% recall**). The model must be used for **human-assisted review**, not automatic quality acceptance. The Pi evaluator calculates AP50/mAP50, not final test-set mAP50–95.

**Deployment verification:** Streamlit upload, defect boxes, on-demand Llama guidance, SQLite saving, and inspection history were exercised after model activation. **42/42 unit tests passed** and `PRAGMA integrity_check` returned `ok` on the local SQLite database (five records at the time). Unit tests emitted nonfatal SQLite connection `ResourceWarning` messages, which remain a cleanup task.

See [the measured technical report](docs/TECHNICAL_REPORT.md), [dataset evaluation details](docs/DATASET_EVALUATION.md), and [submission checklist](docs/SUBMISSION_CHECKLIST.md).

### Run the lightweight logic tests

```bash
cd ~/Industrial_Defect_Detection
.venv/bin/python -m unittest discover -s tests -v
```

These tests do not load model weights, access the internet, or require an attached camera.

## Evaluate a held-out labeled test set

First run the **read-only dataset inventory** to find real annotation files and explicit split information:

```bash
python scripts/audit_dataset.py --roots datasets
```

The project's actual NEU-DET test split is under `datasets/processed/NEU-DET/test`. To reproduce the final improved-model evaluation with the stored annotations, use:

```bash
cd ~/Industrial_Defect_Detection
source .venv/bin/activate
python scripts/evaluate_dataset.py \
  --model models/exported/best_ncnn_model \
  --images datasets/processed/NEU-DET/test/images \
  --labels datasets/processed/NEU-DET/test/labels \
  --data-yaml datasets/processed/NEU-DET/data.yaml \
  --imgsz 320 \
  --operating-conf 0.30 \
  --output logs/evaluation/improved_model_test_final
```

For **YOLO TXT** labels, the `--data-yaml` argument shown above is required to avoid class index mismatches. The evaluator uses dashboard threshold 0.30 for precision/recall/F1 and a lower detection cutoff for AP50. Add `--train-images /path/to/train/images` when available to reject exact-duplicate train/test images; never claim a split is held out solely because it is named `test`.

The evaluator checks that each image has a matching annotation; it refuses to treat missing labels as normal images. Results from the command above are saved under `logs/evaluation/improved_model_test_final/` (including `metrics.json` and `per_class_metrics.csv`). It reports **AP50 only**, not mAP50-95. Six-class mAP50 requires test examples for all six classes.

The final 180-image evaluation has been completed and is summarized above. Preserve the test split for reporting rather than further threshold tuning; avoid lengthy inference runs on inadequate power.

## Final documentation

- [Six-class labeled dataset preparation and evaluation](docs/DATASET_EVALUATION.md)
- [Technical report with final results](docs/TECHNICAL_REPORT.md)
- [Demonstration video script](docs/DEMO_SCRIPT.md)
- [Submission and browser verification checklist](docs/SUBMISSION_CHECKLIST.md)
- Optional WebRTC package versions: [requirements-camera.txt](requirements-camera.txt)

Llama explanation text is saved with annotated upload and camera snapshots. The Pi's deployed inference flow, browser dashboard workflow, dataset-audit unit tests, and database persistence have been verified; all 42 unit tests passed.

## Safety and limitations

- The NEU-DET training/evaluation setup does not establish reliable acceptance of normal (defect-free) surfaces.
- YOLO bounding boxes and confidences are **not** real-world dimensions, defect causes, or calibrated severity.
- Llama can generate incomplete or unsupported advice. The prototype uses a limited output filter and a class-specific rule-based fallback, but neither guarantees factual correctness.
- Human inspection and approved site-specific criteria must determine any production QC disposition.
- The Pi dashboard now has an **experimental** live-camera tab for Pi-connected USB, CSI and IP sources plus optional browser WebRTC. Live-camera hardware operation has not yet been tested, and any separate Windows camera implementation must be evaluated independently.

## Next work

If further development is desired, verify **Pi-connected live-camera hardware**, investigate SQLite connection cleanup warnings, test sustained operation, and improve crazing recall on future training/validation data. Do not tune against the final test set.
