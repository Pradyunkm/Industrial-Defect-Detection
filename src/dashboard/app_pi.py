"""Raspberry Pi local image-upload dashboard. Webcam feature remains in Windows app."""
import csv
import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'inference'))
from common import predict, review  # noqa: E402
from llama_explain import explain_defects  # noqa: E402


def grounded_inspection_guidance(detections, summary):
    """Build class-specific guidance using only YOLO labels."""
    from collections import Counter

    actions = {
        "crazing": "inspect for fine interconnected surface lines",
        "scratches": "inspect for linear surface marks",
        "inclusion": "inspect for possible material inclusions",
        "patches": "inspect for irregular surface patches",
        "pitted_surface": "inspect for possible surface pitting",
        "rolled-in_scale": "inspect for scale-like surface features",
    }

    counts = Counter(str(d.get("class", "unknown"))
                     for d in detections)

    if not counts:
        return (
            summary + " Perform additional visual inspection. "
            "No detections does not establish acceptance."
        )

    checks = [
        f"{count} possible {name} region(s): "
        f"{actions.get(name, 'visually inspect the marked areas')}"
        for name, count in sorted(counts.items())
    ]

    return (
        summary + " Recommended checks: "
        + "; ".join(checks)
        + ". Verify against approved inspection criteria. "
        "No dimensions, causes, or pass/fail decisions are inferred."
    )


DB = ROOT / 'database' / 'inspections.db'
OUT = ROOT / 'logs' / 'annotated'
DEFAULT_MODEL = ROOT / 'models' / 'exported' / 'best_ncnn_model'

st.set_page_config(page_title='PS5 | Raspberry Pi Defect Inspector', layout='wide', page_icon='🔎')
OUT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(DB) as db:
    db.execute('''CREATE TABLE IF NOT EXISTS pi_inspections (
        id TEXT PRIMARY KEY, timestamp_utc TEXT, filename TEXT, model TEXT,
        conf REAL, imgsz INTEGER, detection_count INTEGER,
        priority TEXT, recommendation TEXT, detections_json TEXT,
        annotated_file TEXT)''')

@st.cache_resource(show_spinner='Loading NCNN model on Raspberry Pi CPU…')
def get_model(path):
    return YOLO(path, task='detect')

st.title('🔎 PS5 Industrial Steel Defect Inspector — Raspberry Pi')
st.caption('Offline NCNN inference on Arm CPU · Local SQLite logging · Uploaded images')
model_path = st.sidebar.text_input('NCNN model folder', str(DEFAULT_MODEL))
conf = st.sidebar.slider('Confidence threshold', .05, .95, .30, .05)
imgsz = st.sidebar.select_slider('Image size', [320, 416, 512], value=320)
st.sidebar.warning('Demo QC rules only. Confidence is not severity.')
if not Path(model_path).is_dir():
    st.error(f'Model folder missing: {model_path}. Transfer the entire exported NCNN directory first.')
    st.stop()
try:
    model = get_model(str(Path(model_path).resolve()))
except Exception as err:
    st.error(f'Model could not load: {err}')
    st.stop()

upload_tab, history_tab = st.tabs(['📤 Upload image', '🗃️ Inspection history'])
with upload_tab:
    file = st.file_uploader('Upload a steel-surface JPG or PNG', type=['jpg','jpeg','png','bmp'])
    if file:
        frame = cv2.imdecode(np.frombuffer(file.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error('Invalid image')
        else:
            with st.spinner('Inspecting locally on the Pi CPU…'):
                annotated, detections, ms = predict(frame, model, conf=conf, imgsz=imgsz)
            a, b = st.columns(2)
            with a:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Original image', width='stretch')
            with b:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption='Annotated output', width='stretch')
            priority, advice = review(detections)
            st.write(f'**Detected regions:** {len(detections)} | **Wall-clock inference:** {ms:.0f} ms | **Review:** {priority}')
            st.caption(advice)
            if detections:
                st.dataframe(detections, hide_index=True, width='stretch')
            if st.button('🤖 Explain with Llama (offline)', key='pi_llama_button'):
                with st.spinner('Generating locally with ExecuTorch…'):
                    try:
                        report = explain_defects(detections)
                    except Exception as err:
                        st.error(f'Llama explanation unavailable: {err}')
                    else:
                        st.markdown('**YOLO summary**')
                        st.write(report['summary'])
                        advice = report['llama_explanation'].strip()
                        if (len(advice) < 60 or
                                not advice.endswith(('.', '!', '?')) or
                                'use the following format' in advice.lower()):
                            advice = grounded_inspection_guidance(detections, report['summary'])
                            st.caption('Incomplete Llama response; YOLO-based inspection guidance shown.')
                        st.markdown('**Inspection guidance**')
                        st.write(advice)
                        st.caption('Advisory only. Human verification required.')
            ok, encoded = cv2.imencode('.png', annotated)
            if ok:
                png = encoded.tobytes()
                st.download_button('⬇️ Download annotated PNG', png, file_name='annotated_'+Path(file.name).stem+'.png', mime='image/png')
                if st.button('💾 Save inspection to SQLite'):
                    id_ = uuid.uuid4().hex[:12]
                    location = OUT / (id_+'.png')
                    location.write_bytes(png)
                    try:
                        with sqlite3.connect(DB) as db:
                            db.execute('INSERT INTO pi_inspections VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                              (id_, datetime.now(timezone.utc).isoformat(), file.name, model_path,
                               conf, imgsz, len(detections), priority, advice, json.dumps(detections), str(location)))
                    except Exception:
                        location.unlink(missing_ok=True)
                        raise
                    st.success(f'Inspection {id_} saved locally.')
with history_tab:
    with sqlite3.connect(DB) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT * FROM pi_inspections ORDER BY timestamp_utc DESC LIMIT 300')]
    if not rows:
        st.info('No saved inspections yet.')
    else:
        cols = ['timestamp_utc','filename','detection_count','priority','recommendation']
        st.dataframe([{k:r[k] for k in cols} for r in rows], width='stretch', hide_index=True)
        csv_data = io.StringIO()
        writer = csv.DictWriter(csv_data, fieldnames=cols)
        writer.writeheader()
        writer.writerows([{k:r[k] for k in cols} for r in rows])
        st.download_button('⬇️ Export CSV', csv_data.getvalue(), file_name='pi_inspections.csv', mime='text/csv')
        idx = st.selectbox('View saved image', list(range(len(rows))),
                           format_func=lambda i: f"{rows[i]['timestamp_utc']} | {rows[i]['filename']}")
        saved = Path(rows[idx]['annotated_file'])
        if saved.is_file():
            st.image(str(saved), width='stretch')
        st.json(json.loads(rows[idx]['detections_json']))
st.divider()
st.warning('Research prototype only. NEU-DET lacks verified normal examples. No detections does not prove quality acceptance.')
