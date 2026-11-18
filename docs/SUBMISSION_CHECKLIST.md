# PS5 final verification and submission checklist

**Official project deliverables:** Source code, application, demonstration video and technical report. Project must be hosted on GitHub; use the official challenge submission form and check current deadline.

## Completed (supported by observed outputs)

- [x] YOLOv8 Nano exported to NCNN and running on Raspberry Pi 5.
- [x] ExecuTorch Llama 3.2 1B runner working on Pi.
- [x] Combined YOLO → Llama test succeeded with network access blocked.
- [x] SQLite inspection history contained three records during prior test.
- [x] GitHub repository accessible and README updated.
- [x] Multi-camera source selectors added to website.
- [x] User reported Windows browser WebRTC webcam feed working.
- [x] Previous automated tests passed (17/17 before the newest source additions).
- [x] Earlier full verification passed (16/16 excluding power checks).

## Implementation completed in code; verify again on Pi

- [ ] Pull latest changes: `git pull --ff-only origin main`.
- [ ] Run lightweight regression suite: `.venv/bin/python -m unittest discover -s tests -v`.
- [ ] Compile dashboard source: `.venv/bin/python -m py_compile src/dashboard/*.py`.
- [x] Open site in Windows browser: `http://localhost:8502` with SSH local tunnel running (user-reported)
- [x] Upload image; see correct original and annotated views (user-reported)
- [x] Toggle sidebar bbox annotations ON/OFF (user-reported)
- [x] Click Explain with Llama; ensure explanation remains visible after interacting (user-reported)
- [x] Save inspection; confirm History row and annotated image (user-reported)
- [x] Export History CSV; confirm latest saved record is present (user-reported)
- [x] Start browser webcam briefly and confirm overlay labels and detections (user-reported)
- [x] Stop browser webcam; download last annotated frame; request Llama guidance and save the captured inspection (user-reported)
- [ ] Confirm no unexpected Streamlit errors in `logs/streamlit_pi.log` after extended use.
- [ ] Optional: test Pi-connected USB webcam / CSI camera / RTSP stream when hardware is available.

## Quantitative evaluation (not yet performed)

- [ ] Run `python scripts/audit_dataset.py --roots datasets` to locate usable annotations and real split provenance.
- [ ] Locate actual held-out NEU-DET images AND matching ground-truth XML or TXT labels.
- [ ] Run `scripts/evaluate_dataset.py` with correct --images and --labels folders.
- [ ] Inspect generated `logs/evaluation/metrics.json` and per-class CSV.
- [ ] Verify the test set contains all six classes before claiming six-class mAP50.
- [ ] Generate mAP50–95 separately if required; the provided evaluator reports AP50 only.
- [ ] Record latency distribution, CPU/RAM and temperature under stable power.
- [ ] Add evaluation numbers and caveats to the technical report and README.

## Operational reliability

- [ ] Use reliable dedicated Raspberry Pi 5 power. Historical undervoltage/throttling flags have been seen on laptop USB-C power.
- [ ] If power changes, reboot and rerun checks.
- [ ] Validate camera disconnect/reconnect and user permission denial.
- [ ] Recheck stream stability and save results under repeated use.
- [ ] Avoid exposing RTSP camera credentials or network stream URLs in public screenshots/logs.

## Submission readiness

- [ ] Check official problem statement and submission deadline: https://armbharatchallenge.com/
- [ ] Capture architecture illustration and dashboard screenshots.
- [ ] Record the demonstration video following `docs/DEMO_SCRIPT.md`.
- [ ] Produce a final report PDF based on `docs/TECHNICAL_REPORT.md`, with actual metrics and screenshots.
- [ ] Verify GitHub repository is public and source files/README accessible.
- [ ] Confirm model artifacts are not exposed if licensing or size restricts distribution.
- [ ] Provide model provenance and separate deployment instructions.
- [ ] Submit repository, application demonstration, video and report in the official form.

## Quick commands

```bash
cd ~/Industrial_Defect_Detection
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m py_compile src/dashboard/*.py src/inference/*.py scripts/*.py
```

For heavy real-model verification, first ensure reliable power; **do not repeatedly rerun inference simply to clear historical throttling flags**.

For a dataset smoke test using actual labeled files (example directories only):

```bash
python scripts/evaluate_dataset.py \
  --images datasets/processed/test/images \
  --labels datasets/processed/test/labels \
  --limit 10
```
