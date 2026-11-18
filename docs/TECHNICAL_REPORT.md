# Technical report — Industrial Defect Detection & Quality-Control Assistant

**Bharat AI-SoC Challenge 2026–27 | PS5 — Manufacturing & Industry 4.0**  
**Status:** Phase 1 technical-report draft; update evaluation tables and screenshots before submission.  
**Repository:** https://github.com/Pradyunkm/Industrial-Defect-Detection  
**Target:** Raspberry Pi 5 (8 GB RAM), ARM64, CPU-only

## 1. Executive summary

This prototype performs edge-based visual inspection of steel surfaces using YOLOv8 Nano exported to NCNN on Raspberry Pi 5. It identifies possible surface defect regions, overlays detection boxes, and provides advisory review categories. The operator can optionally request a short, on-device explanation from Llama 3.2 1B Instruct running through ExecuTorch. The Streamlit dashboard supports uploaded images, browser-camera snapshots, experimental live feeds from multiple camera types, annotated-image downloads, SQLite inspection history, and CSV export.

All inference is designed to run locally without a cloud API. A combined YOLO → Llama inference test was successfully conducted inside a network namespace that could not reach external IP addresses. The project is a research prototype; no measurement, acceptance decision, calibrated severity or industrial certification is claimed.

## 2. Problem and objective

Manual steel-surface inspection may be inconsistent, time-consuming and difficult to document. PS5 asks for on-device visual defect and anomaly detection, operator assistance, logging and real-time edge operation. The project focuses on small-device, offline processing and audit-friendly inspection records.

**Implemented:** six-class detector, optional Llama advice, basic human-review prompts, timestamped inspection history and annotated evidence.  
**Not yet established:** true production-rate throughput, verified defect severity labels, accuracy on fully held-out annotated data, or industrial false-positive/false-negative rates.

## 3. Hardware and software

| Component | Prototype configuration |
|---|---|
| Compute | Raspberry Pi 5, 8 GB, ARM64 CPU |
| Operating environment | Debian/Raspberry Pi Linux, Python 3.13 virtual environment |
| Detector | Ultralytics YOLOv8 Nano, NCNN export |
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
               YOLOv8 Nano -> NCNN CPU
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

## 7. Verification evidence already observed

| Verification | Recorded outcome |
|---|---|
| Combined isolated YOLO → Llama | Completed successfully |
| Network egress in isolated namespace | Blocked; `[Errno 101] Network is unreachable` |
| Example image | `scratches_271.jpg` |
| Detections on example | 3 boxes, all classified as scratches |
| Confidence scores | 0.641, 0.568, 0.432 |
| YOLO inference in one run | 1,416 ms |
| YOLO inference in a subsequent run | 3,668 ms |
| Llama response | Generated a short visual inspection recommendation |
| SQLite historical inspections | 4 saved inspections in the latest full check |
| Latest comprehensive verification | 18/18 passed with power checks deliberately excluded; actual isolated inference and SQLite/CSV workflow verified |
| Latest confirmed regression suite | 32/32 passed before new dataset-audit tests were added |
| Browser workflow | User confirmed all 8 manual browser checks passed; no certified FPS or accuracy measurement |

These are **single-test observations, not a benchmark dataset**. The newly added read-only dataset audit and extended evaluation guards still require regression tests on the Pi. A response from the Streamlit health endpoint is not proof that each browser control works.

## 8. Quantitative evaluation plan and missing results

Use an annotated **held-out** subset of NEU-DET not used in training. The script accepts Pascal VOC XML or YOLO TXT bounding-box annotations, matches detections to ground truth at IoU >= 0.50 and generates per-class precision, recall, F1 and AP50, plus mAP50 over the classes actually represented.

```bash
cd ~/Industrial_Defect_Detection
source .venv/bin/activate
python scripts/evaluate_dataset.py \
  --images datasets/processed/test/images \
  --labels datasets/processed/test/labels \
  --model models/exported/best_ncnn_model \
  --output logs/evaluation
```

**Dataset folder names are examples:** inspect the available Pi test-set layout before executing. The script rejects images missing annotation files rather than silently treating them as defect-free. It does not calculate mAP50–95. The results in `logs/evaluation/metrics.json` and `per_class_metrics.csv` must be generated from actual annotated test data.

| Metric | Status |
|---|---|
| Six-class precision / recall / F1 | Awaiting evaluation run |
| Per-class AP50 | Awaiting evaluation run |
| mAP50, all six classes | Awaiting adequate held-out representation |
| mAP50–95 | Not measured by this script |
| End-to-end webcam FPS | Not measured |
| Peak RAM / CPU and sustained load | Not measured |

Do not add a guessed accuracy number to the submission.

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

## 11. Next engineering work

1. Run the new dataset-audit tests. The prior 32 regression tests and the eight-step manual browser walkthrough passed.
2. Collect a true held-out annotated test set and run the per-class evaluation script.
3. Run a short and repeatable performance benchmark on stable power.
4. Test at least one real Pi-connected USB or CSI camera if available.
5. Capture proof screenshots and a narrated demonstration video.
6. Export this draft as a PDF with real evaluation results and submit with the public repository.

## 12. Project deliverables

- **Source code:** public GitHub repository linked above.
- **Application:** on-device Streamlit dashboard served by Raspberry Pi.
- **Demonstration video:** to be recorded; see `docs/DEMO_SCRIPT.md`.
- **Technical report:** this draft; replace unmeasured fields and export to PDF for submission.

**Reference:** [Official Bharat AI-SoC Challenge PS5 specification](https://armbharatchallenge.com/) — required Llama/ExecuTorch technologies and four deliverables.
