# Industrial Defect Detection & Quality-Control Assistant

**Bharat AI-SoC Challenge 2026–27 · Problem Statement 5**

An **offline-capable steel-surface inspection research prototype** running on a Raspberry Pi 5. It combines YOLOv8 Nano (exported to NCNN) for image-based defect detection with Llama 3.2 1B Instruct (via ExecuTorch) for optional, on-device inspection advice.

> **Research prototype, not a certified quality-control system.** Detection confidence is not defect severity; no detections do not prove a defect-free surface. Human verification is required.

## System architecture

```text
Uploaded steel image
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

Llama receives a **text summary of YOLO detections**, not the image pixels. It is invoked by the user, not on every camera frame. The Raspberry Pi dashboard supports **image uploads**; the separate Windows development application contains live-camera functionality and is not included in this repository.

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
  dashboard/app_pi.py       Streamlit upload and inspection-history UI
  inference/common.py       YOLO inference and demo review rules
  inference/infer_image.py  Single-image command-line inference
  inference/llama_explain.py  Optional offline Llama explanation helper
README.md
requirements.txt
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

In the **Upload image** tab, upload a steel-surface JPG or PNG, review the YOLO detections, optionally choose **Explain with Llama (offline)**, and download the annotated PNG or save the inspection to SQLite. The **Inspection history** tab displays saved records and supports CSV export.

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
- **Separate network-isolated inference tests** for YOLO and Llama
- Combined YOLO → Llama explanations inside the Streamlit dashboard
- SQLite inspection saving, annotated-file persistence, and CSV generation/download

**Still to verify:** the full dashboard workflow under network isolation, systematic evaluation across the six defect classes, and repeatable performance under a stable dedicated power supply. No overall accuracy or real-time camera-performance claim is made here.

## Safety and limitations

- The NEU-DET training/evaluation setup does not establish reliable acceptance of normal (defect-free) surfaces.
- YOLO bounding boxes and confidences are **not** real-world dimensions, defect causes, or calibrated severity.
- Llama can generate incomplete or unsupported advice. The prototype uses a limited output filter and a class-specific rule-based fallback, but neither guarantees factual correctness.
- Human inspection and approved site-specific criteria must determine any production QC disposition.
- The uploaded-image Pi dashboard does not provide a tested live-camera pipeline; any separate Windows camera implementation must be evaluated independently.

## Next work

Expand evaluation results and reproducibility instructions, test full offline integration, and prepare architecture screenshots and a demonstration video for the challenge submission.
