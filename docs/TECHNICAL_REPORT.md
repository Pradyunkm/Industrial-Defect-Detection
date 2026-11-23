# Technical report — Industrial Defect Detection & Quality-Control Assistant

**Bharat AI-SoC Challenge 2026–27 | PS5 — Manufacturing & Industry 4.0**  
**Status:** Updated with November 23, 2026 model evaluation and integration results; final submission review pending.
**Repository:** https://github.com/Pradyunkm/Industrial-Defect-Detection  
**Target:** Raspberry Pi 5 (8 GB RAM), ARM64, CPU-only

## 1. Executive summary

This prototype performs edge-based visual inspection of steel surfaces using an improved 100-epoch YOLO detector exported to NCNN at 320 pixels on Raspberry Pi 5. It identifies possible surface defect regions, overlays detection boxes, and provides advisory review categories. The operator can optionally request a short, on-device explanation from Llama 3.2 1B Instruct running through ExecuTorch. The Streamlit dashboard supports uploaded images, browser-camera snapshots, experimental live feeds from multiple camera types, annotated-image downloads, SQLite inspection history, and CSV export.

All inference is designed to run locally without a cloud API. A combined YOLO → Llama inference test was successfully conducted inside a network namespace that could not reach external IP addresses. The project is a research prototype; no measurement, acceptance decision, calibrated severity or industrial certification is claimed.

## 2. Problem and objective

Manual steel-surface inspection may be inconsistent, time-consuming and difficult to document. PS5 asks for on-device visual defect and anomaly detection, operator assistance, logging and real-time edge operation. The project focuses on small-device, offline processing and audit-friendly inspection records.

**Implemented:** six-class detector, optional Llama advice, basic human-review prompts, timestamped inspection history and annotated evidence.  
**Not yet established:** production-rate camera throughput, calibrated defect severity, and accuracy on independent industrial production data.

## 3. Hardware and software

| Component | Prototype configuration |
|---|---|
| Compute | Raspberry Pi 5, 8 GB, ARM64 CPU |
| Operating environment | Debian/Raspberry Pi Linux, Python 3.13 virtual environment |
| Detector | 100-epoch YOLO, 11.13M parameters, NCNN 320px; original YOLOv8 Nano retained as backup |
| Inspection dataset | NEU-DET steel surface defect classes |
| Advisory model | Llama 3.2 1B Instruct .pte |
| LLM runtime | ExecuTorch 1.5.0, ARM64 native build |
| Operator interface | Streamlit |
| Image processing | OpenCV, NumPy |
| Persistence | Local SQLite and annotated PNG images |
| Live video | Browser WebRTC, USB/V4L2, Raspberry Pi CSI, RTSP/HTTP; hardware-specific validation pending |
| Network requirement for inference | None for the verified local model paths |

The Pi project `.venv` and the ExecuTorch native installation are separate. Model artifacts and datasets are not included in the public repository.

## 4. Architecture and data flow

```text
Uploaded image / browser webcam / Pi-side USB / CSI / IP camera
                          |
                   OpenCV BGR frame
                          |
               100-epoch YOLO -> NCNN CPU
                          |
              Boxes, class, confidence
                     /         \
         Streamlit overlay     Demo review rules
            and download           |
                     \            /
                       Structured findings
                               |
                      Optional Llama request
                               |
                  ExecuTorch Llama 3.2 1B
                               |
                  Short visual-check guidance
                               |
                     Human QC operator
                               |
              SQLite timestamp + annotated PNG
                               |
                    History + CSV export
```

Llama does **not** inspect image pixels directly. It receives a short grounded text summary derived from YOLO labels, and is asked not to invent dimensions, causes, severity or pass/fail outcomes. A basic filter and class-specific fallback limit some unsuitable responses, but cannot guarantee correctness.

## 5. Classes and review outputs

The six predicted classes are: crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches.

YOLO reports bounding boxes and confidence scores. The app's review outputs are **UNVERIFIED**, **REVIEW**, **MEDIUM REVIEW**, and **HIGH REVIEW** based on example rules about class and relative bounding-box area. These are prioritization prompts for a human, **not calibrated severity grades**. No detection does not indicate that a component is accepted.

## 6. Implementation overview

- `src/inference/common.py`: NCNN prediction wrapper and review prompts.
- `src/inference/llama_explain.py`: subprocess invocation of the ExecuTorch `llama_main` runner.
- `src/inference/inspection_store.py`: SQLite, annotated PNG and CSV operations.
- `src/dashboard/app_pi.py`: upload, camera selection, captured live-frame review, and history.
- `src/dashboard/browser_live.py`: optional laptop/phone browser webcam via WebRTC.
- `src/dashboard/camera_sources.py`: USB, Pi CSI and network camera adapters.
- `scripts/evaluate_dataset.py`: held-out evaluation from VOC XML or YOLO TXT labels.
- `scripts/audit_dataset.py`: read-only dataset inventory and explicit split audit.
- `tests/`: unit, parser and camera-mock tests.

## 7. Verification evidence

| Verification | Confirmed result |
|---|---|
| Platform | Raspberry Pi 5, CPU-only NCNN inference |
| Deployed detector | Improved 100-epoch YOLO, exported at 320x320 |
| Original detector | Preserved in a separate backup |
| Automated tests | 42/42 passed |
| SQLite integrity | OK; 5 inspection records during verification |
| Dashboard | Image upload, detection overlay, and inspection guidance displayed |
| Inspection persistence | Save to SQLite and inspection history demonstrated |
| Offline Llama | Earlier network-isolated YOLO-to-Llama workflow passed; guidance also displayed after deployment |
| Known warning | Python ResourceWarning for unclosed SQLite connections during unit tests |

The improved model was successfully loaded from
`models/exported/best_ncnn_model` and tested through the
Streamlit dashboard. The dashboard was accessed through an
SSH tunnel from Windows.

## 8. Quantitative model evaluation

The NEU-DET dataset uses six defect classes. The original
training dataset contains 1440 training, 180 validation,
and 180 test images. The test set has 446 annotated defect
regions. Earlier checks found no identical filenames or
byte-identical images shared between training/validation
and test splits.

The original training configuration and split were checked,
but the Pi evaluator printed a provenance warning because
the optional --train-images argument was not provided.
These checks do not exclude all possible dataset leakage.

### Final Raspberry Pi test-set comparison

Both NCNN models were evaluated on the same 180 annotated
test images at IoU 0.50 and operating confidence 0.30.

| Metric | Original YOLOv8 Nano | Improved 100-epoch YOLO |
|---|---:|---:|
| mAP50 | 74.32% | 74.66% |
| True positives | 268 | 286 |
| False positives | 73 | 67 |
| False negatives | 178 | 160 |
| Micro F1 at confidence 0.30 | 68.11% | 71.59% |
| Average Pi prediction latency | 76.5 ms | 92.2 ms |

Latency was measured over 12 sequential predictions per
model. It is not an end-to-end live-camera FPS measurement.

### Final per-class AP50

| Defect | Original | Improved |
|---|---:|---:|
| Crazing | 44.61% | 46.03% |
| Inclusion | 79.55% | 81.50% |
| Patches | 92.16% | 92.78% |
| Pitted surface | 73.11% | 81.61% |
| Rolled-in scale | 67.78% | 64.02% |
| Scratches | 88.71% | 82.00% |

The improved model gives better overall detection F1 at
the selected operating threshold but is slower. It also
performs worse in AP50 for rolled-in scale and scratches.

Crazing recall on the final test set is only 21.52% at
confidence 0.30. This is a significant limitation and
requires human quality-control review.

The Pi evaluation script reports AP50/mAP50 only. A separate
matched Windows validation run measured mAP50-95 at 45.41%
for the improved PyTorch model at 320 pixels; this is not
a final test-set mAP50-95 score.

The six-class defect-only dataset does not establish
reliable acceptance of normal, defect-free materials.

## 9. Edge deployment and offline verification

The YOLO inference engine is NCNN on the Raspberry Pi CPU. The Llama model is stored as a `.pte` file and executed locally using ExecuTorch. Network isolation was tested using an isolated Linux network namespace and confirmed unreachable TCP connection to `1.1.1.1:443`. This supports the claim that combined inference does not require internet access. It does **not** yet verify that the complete multiuser Streamlit browser session works under all network restrictions.

## 10. Known risks, limits and safety

- The device previously reported `throttled=0x50000`, indicating historical undervoltage/throttling flags; a stable dedicated Pi 5 power supply is recommended before sustained live testing.
- The NEU-DET workflow does not establish reliable normal/defect-free component acceptance.
- Detection confidence and bounding-box area are not physical severity or component dimensions.
- The camera adapters have been unit-tested with mocks; specific USB/CSI/IP camera hardware must still be verified.
- Browser WebRTC uses camera permission in a secure browser context (HTTPS or localhost). It should not be exposed as an unauthenticated production service.
- Llama outputs require human review; safety text filtering is limited.
- Tests on monitor-displayed sample images are not a substitute for a real industrial camera evaluation.

## 11. Remaining engineering work

1. Capture demonstration screenshots and video.
2. Verify the final submission requirements and deadline.
3. Document model provenance and installation instructions.
4. Measure sustained camera FPS and memory use if hardware is available.
5. Investigate SQLite connection-cleanup warnings.
6. Improve crazing recall in future model experiments without tuning
   on the final test set.

## 12. Project deliverables

- **Source code:** public GitHub repository linked above.
- **Application:** on-device Streamlit dashboard served by Raspberry Pi.
- **Demonstration video:** to be recorded; see `docs/DEMO_SCRIPT.md`.
- **Technical report:** updated with measured results; complete final review and export to PDF for submission.

**Reference:** [Official Bharat AI-SoC Challenge PS5 specification](https://armbharatchallenge.com/) — required Llama/ExecuTorch technologies and four deliverables.
