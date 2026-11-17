# PS5 demonstration video — recording plan

**Goal:** Show a real Raspberry Pi 5 running YOLO NCNN + offline ExecuTorch Llama, with an upload flow, browser webcam and inspection logging. Suggested length: **3–4 minutes** (adjust to any official submission rules).

## Before recording

- Connect Raspberry Pi 5 and Windows laptop to the same network.
- Prefer a stable dedicated power source; avoid long load tests on laptop USB-C power.
- On the Pi, run `source .venv/bin/activate` from the repository and confirm Streamlit is available at port 8501.
- For the Windows laptop browser camera, keep the SSH local tunnel active: `ssh -N -L 8502:127.0.0.1:8501 pi@172.16.37.63`. **The Pi IP may change after reboot.**
- Open `http://localhost:8502` in the Windows browser.
- Prepare one sample image of scratches and a different defect class, if available.
- Open `docs/TECHNICAL_REPORT.md` and `logs/evaluation/metrics.json` if actual metrics have been generated.
- Record with a Windows screen recorder; show the Pi device and terminal for proof of on-device execution.
- Do not expose private SSH keys, tokens, other browser tabs, personal details, or camera views of bystanders.

## Narration and screen plan

| Approximate position | Show | Narration |
|---|---|---|
| 0:00–0:20 | Project title and Pi 5 hardware | "This is an offline-capable steel defect inspection assistant on Raspberry Pi 5 for Bharat AI-SoC PS5." |
| 0:20–0:45 | Architecture diagram and model files | "YOLOv8 Nano runs through NCNN for visual detection. Llama 3.2 1B Instruct runs via ExecuTorch on the Pi CPU to suggest a visual check." |
| 0:45–1:30 | Upload tab: select a steel image, show before/after annotation, boxes and confidence | "The application detects possible defect regions; confidence is not physical severity and all findings require human verification." |
| 1:30–2:05 | Live Camera → Browser webcam; press START briefly; show YOLO overlay and counts | "The browser camera sends frames to the Raspberry Pi CPU. Frame updates depend on inference latency, not a guaranteed 30 FPS." |
| 2:05–2:35 | STOP feed, select a captured image, request optional Llama explanation | "The language model receives YOLO text labels and produces short inspection guidance locally." |
| 2:35–3:05 | Save the captured inspection, open History, show annotated image and export CSV | "Records, screenshots and recommendations are stored in local SQLite; no cloud inference is used." |
| 3:05–3:35 | Show offline test output, repository README and test results | "Combined inference passed with external connectivity blocked. Unit tests and reproducible setup are published." |
| 3:35–3:50 | Limitations and closing | "This is a research prototype. Held-out accuracy evaluation and industrial severity calibration are separate required validation steps." |

**Note:** Llama inference can be slow on the CPU. Do not label edited footage as real-time inference unless it is shown at actual speed. If time-lapsing, mark it visibly.

## Evidence screenshots to capture

1. Upload tab: original and annotated steel image side by side.
2. Live Camera source dropdown: USB, Pi CSI, RTSP/HTTP and browser webcam.
3. Browser WebRTC camera preview with YOLO boxes and detection table.
4. Llama summary and visual-inspection guidance.
5. SQLite Inspection History including the saved captured image.
6. Downloaded annotated PNG and CSV file (avoid sensitive metadata).
7. Terminal showing an offline inference test and unit-test pass summary.
8. Optional actual Pi USB/CSI camera test, only if hardware is available and verified.

## What to avoid claiming

- Do not claim calibrated defect severity, certified pass/fail or measurement in millimetres.
- Do not claim 30 FPS or a specific mAP without a verified benchmark.
- Do not claim USB/CSI hardware was tested if only browser webcam was demonstrated.
- Do not imply that Llama directly sees the camera image: it processes text derived from YOLO detections.

## Deliverables

- Video file or externally accessible video link (upload and permissions handled by the team).
- `README.md` and source-code link.
- Technical-report PDF with measured results and screenshot evidence.
- Final checklist in `docs/SUBMISSION_CHECKLIST.md`.
