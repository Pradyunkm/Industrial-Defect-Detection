"""Run all safe Raspberry Pi project checks in a single command.

Does not touch live-camera code, install packages, or alter model weights.
"""
import getpass
import csv
import io
import tempfile
import importlib.metadata as metadata
import os
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
NCNN = ROOT / "models/exported/best_ncnn_model"
IMAGE = ROOT / "datasets/raw/scratches_271.jpg"
ET = Path.home() / "executorch-deployment"
RUNNER = ET / "executorch/cmake-out/examples/models/llama/llama_main"
MODEL = ET / "models/llama3.2-1b/Llama-3.2-1B-Instruct.pte"
TOKENIZER = ET / "models/llama3.2-1b/tokenizer.model"
checks = []


def record(label, ok, detail=""):
    checks.append((label, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}", flush=True)


def run(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, check=False, **kwargs)


def offline_child():
    """Invoked by the parent inside a network namespace as user pi."""
    print("=== NETWORK-ISOLATED INFERENCE ===", flush=True)
    try:
        connection = socket.create_connection(("1.1.1.1", 443), timeout=3)
    except OSError as error:
        print(f"PASS: Internet inaccessible ({error})", flush=True)
    else:
        connection.close()
        print("FAIL: Internet is reachable", flush=True)
        return 1

    sys.path.insert(0, str(ROOT / "src/inference"))
    import cv2
    from ultralytics import YOLO
    from common import predict, review
    from llama_explain import explain_defects
    from inspection_store import save_inspection, load_inspections, export_inspections_csv

    image = cv2.imread(str(IMAGE))
    if image is None:
        print("FAIL: Cannot open sample image", flush=True)
        return 1
    model = YOLO(str(NCNN), task="detect")
    annotated, detections, elapsed = predict(image, model, conf=0.30, imgsz=320)
    print(f"YOLO: {len(detections)} region(s), {elapsed:.0f} ms", flush=True)
    for item in detections:
        print(f"  {item['class']}: {item['confidence']:.3f}", flush=True)
    print(f"Review: {review(detections)[0]}", flush=True)
    if not any(item["class"] == "scratches" for item in detections):
        print("FAIL: No scratches prediction in known sample", flush=True)
        return 1

    report = explain_defects(detections)
    print(f"YOLO summary: {report['summary']}", flush=True)
    print(f"Llama guidance: {report['llama_explanation']}", flush=True)
    if not report["llama_explanation"].strip():
        print("FAIL: Empty Llama guidance", flush=True)
        return 1
    print("PASS: YOLO -> Llama worked with no internet", flush=True)

    # Run one complete REAL inference-output workflow in a temporary directory.
    # No production database or saved inspection is touched.
    print("\n=== ANNOTATION -> SQLITE -> HISTORY -> CSV ===", flush=True)
    image_ok, png = cv2.imencode(".png", annotated)
    if not image_ok:
        print("FAIL: Could not encode annotated PNG", flush=True)
        return 1

    with tempfile.TemporaryDirectory(prefix="pi_inspection_check_") as folder:
        base = Path(folder)
        inspection_id = save_inspection(
            base / "inspections.db", base / "annotated",
            filename=IMAGE.name, model=str(NCNN),
            conf=0.30, imgsz=320, detections=detections,
            priority=review(detections)[0],
            recommendation=report["llama_explanation"],
            annotated_png=png.tobytes(),
        )
        rows = load_inspections(base / "inspections.db")
        csv_rows = list(csv.DictReader(
            io.StringIO(export_inspections_csv(rows))
        ))
        good = (
            len(rows) == 1
            and len(csv_rows) == 1
            and rows[0]["id"] == inspection_id
            and rows[0]["detection_count"] == len(detections)
            and rows[0]["recommendation"] == report["llama_explanation"]
            and csv_rows[0]["filename"] == IMAGE.name
            and csv_rows[0]["recommendation"] == report["llama_explanation"]
            and (base / "annotated" / (inspection_id + ".png")).read_bytes()
                == png.tobytes()
        )
        if not good:
            print("FAIL: PNG, SQLite, history or CSV mismatch", flush=True)
            return 1

    print("PASS: Annotated PNG, SQLite, inspection history, CSV export", flush=True)
    print("PASS: Only temporary test data used; no production records changed", flush=True)
    return 0


def power_check(label):
    try:
        result = run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5)
        value = result.stdout.strip()
        ok = result.returncode == 0 and value == "throttled=0x0"
        record(label, ok, value)
        temperature = run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=5)
        print("  " + temperature.stdout.strip(), flush=True)
        return ok
    except (OSError, subprocess.TimeoutExpired) as error:
        record(label, False, str(error))
        return False


def dashboard_check():
    url = "http://127.0.0.1:8501/_stcore/health"

    def healthy():
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return response.status == 200 and response.read().strip() == b"ok"
        except (OSError, ValueError):
            return False

    if healthy():
        record("Streamlit health endpoint", True, "Port 8501 already responds")
        return

    log = ROOT / "logs/streamlit_pi.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print("Dashboard not responding; launching it on port 8501...", flush=True)
    try:
        with log.open("ab") as handle:
            process = subprocess.Popen(
                [str(PYTHON), "-m", "streamlit", "run",
                 "src/dashboard/app_pi.py", "--server.address", "0.0.0.0",
                 "--server.port", "8501", "--server.headless", "true"],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        for _ in range(15):
            if healthy():
                record("Streamlit health endpoint", True, "Dashboard now responds")
                return
            if process.poll() is not None:
                break
            time.sleep(1)
    except OSError as error:
        record("Streamlit health endpoint", False, str(error))
        return
    record("Streamlit health endpoint", False, f"Check {log}")


def main():
    print("=== COMPLETE RASPBERRY PI VERIFICATION ===", flush=True)
    skip_power = "--skip-power" in sys.argv
    if skip_power:
        print("Power checks deliberately excluded (--skip-power).", flush=True)
        print("CAUTION: Combined inference still runs once; use a stable power source.", flush=True)
        safe_power = True
    else:
        safe_power = power_check("Power before inference")
    result = run(["git", "status", "--porcelain"], capture_output=True, text=True)
    record("Git working tree clean", result.returncode == 0 and not result.stdout.strip(),
           result.stdout.strip() or "clean")

    print("\n=== INSTALLED DEPENDENCIES ===", flush=True)
    expected = {
        "ultralytics": "8.4.211", "streamlit": "1.67.0",
        "numpy": "2.5.4", "opencv-python": "5.0.0.93",
        "ncnn": "1.0.20261020", "torch": "2.15.0",
    }
    for package, desired in expected.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "missing"
        record(package, actual == desired, f"{actual} (reference {desired})")

    print("\n=== MODEL AND TEST FILES ===", flush=True)
    files = {
        "YOLO NCNN directory": NCNN,
        "Sample steel image": IMAGE,
        "ExecuTorch runner": RUNNER,
        "Llama model": MODEL,
        "Llama tokenizer": TOKENIZER,
    }
    file_status = []
    for name, path in files.items():
        found = path.is_dir() if name == "YOLO NCNN directory" else path.is_file()
        record(name, found, str(path))
        file_status.append(found)

    print("\n=== AUTOMATED LOGIC TESTS ===", flush=True)
    unit = run([str(PYTHON), "-m", "unittest", "discover", "-s", "tests", "-v"])
    record("Unit tests", unit.returncode == 0)

    print("\n=== DASHBOARD AND EVALUATION SOURCE CHECK ===", flush=True)
    from glob import glob
    sources = []
    for pattern in ("src/dashboard/*.py", "src/inference/*.py", "scripts/*.py"):
        sources.extend(glob(str(ROOT / pattern)))
    valid_source = bool(sources)
    for filename in sources:
        try:
            with open(filename, "rb") as handle:
                compile(handle.read(), filename, "exec")
        except (OSError, SyntaxError) as error:
            print(f"  Error in {filename}: {error}", flush=True)
            valid_source = False
    record("Python source compilation", valid_source, f"{len(sources)} project files")

    print("\n=== BROWSER WEBCAM DEPENDENCIES ===", flush=True)
    try:
        webrtc_version = metadata.version("streamlit-webrtc")
        av_version = metadata.version("av")
        record("WebRTC and PyAV dependencies", True,
               f"streamlit-webrtc {webrtc_version}, av {av_version}")
    except metadata.PackageNotFoundError:
        print("SKIP: WebRTC optional dependencies unavailable; other camera options are unaffected.", flush=True)

    print("\n=== FULL OFFLINE YOLO + LLAMA ===", flush=True)
    if safe_power and all(file_status):
        try:
            isolated = run(
                ["sudo", "unshare", "--net", "--", "runuser", "-u",
                 getpass.getuser(), "--", str(PYTHON),
                 str(Path(__file__).resolve()), "--isolated"],
                timeout=240,
            )
            record("Network-isolated combined inference", isolated.returncode == 0)
        except (OSError, subprocess.TimeoutExpired) as error:
            record("Network-isolated combined inference", False, str(error))
    else:
        record("Network-isolated combined inference", False,
               "Skipped because power or a required file needs attention")

    print("\n=== INSPECTION DATABASE ===", flush=True)
    db = ROOT / "database/inspections.db"
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
            count = connection.execute("SELECT COUNT(*) FROM pi_inspections").fetchone()[0]
        record("SQLite inspection history", True, f"{count} saved inspections")
    except sqlite3.Error as error:
        record("SQLite inspection history", False, str(error))

    print("\n=== STREAMLIT SERVER ===", flush=True)
    dashboard_check()
    if not skip_power:
        power_check("Power after inference")
    passed = sum(ok for _, ok in checks)
    print(f"\n=== RESULTS: {passed}/{len(checks)} CHECKS PASSED ===", flush=True)
    print("Manual browser check still needed: upload JPG, explain with Llama, "
          "save inspection, review history, export CSV.", flush=True)
    print("Camera code and model files were not modified.", flush=True)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    try:
        status = offline_child() if "--isolated" in sys.argv else main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        status = 1
    sys.exit(status)
