"""Raspberry Pi dashboard: image uploads, optional Pi-side live camera, inspection history."""
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
from inspection_store import (  # noqa: E402
    init_db, save_inspection, load_inspections, export_inspections_csv,
)
from camera_sources import CameraOpenError, camera_inventory, open_camera  # noqa: E402
from browser_live import render_browser_live  # noqa: E402


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
init_db(DB)

@st.cache_resource(show_spinner='Loading NCNN model on Raspberry Pi CPU…')
def get_model(path):
    return YOLO(path, task='detect')

st.title('🔎 PS5 Industrial Steel Defect Inspector — Raspberry Pi')
st.caption('Offline NCNN inference on Arm CPU · Image uploads · Optional camera feed · Local SQLite')
model_path = st.sidebar.text_input('NCNN model folder', str(DEFAULT_MODEL))
conf = st.sidebar.slider('Confidence threshold', .05, .95, .30, .05)
imgsz = st.sidebar.select_slider('Image size', [320, 416, 512], value=320)
show_annotations = st.sidebar.checkbox('Show YOLO bounding-box annotations', value=True)
st.sidebar.warning('Demo QC rules only. Confidence is not severity.')
if not Path(model_path).is_dir():
    st.error(f'Model folder missing: {model_path}. Transfer the entire exported NCNN directory first.')
    st.stop()
try:
    model = get_model(str(Path(model_path).resolve()))
except Exception as err:
    st.error(f'Model could not load: {err}')
    st.stop()

upload_tab, live_tab, history_tab = st.tabs(['📤 Upload image', '📹 Live camera', '🗃️ Inspection history'])
with upload_tab:
    uploaded_file = st.file_uploader('Upload a steel-surface JPG or PNG', type=['jpg','jpeg','png','bmp'])
    st.caption('Optional browser camera snapshot (not a continuous live feed; HTTPS or localhost may be required).')
    browser_photo = st.camera_input('Capture a still photo from your browser camera', key='pi_browser_snapshot')
    file = uploaded_file if uploaded_file is not None else browser_photo
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
                st.image(cv2.cvtColor(annotated if show_annotations else frame, cv2.COLOR_BGR2RGB), caption='Annotated output' if show_annotations else 'Original (annotations hidden)', width='stretch')
            priority, advice = review(detections)
            st.write(f'**Detected regions:** {len(detections)} | **Wall-clock inference:** {ms:.0f} ms | **Review:** {priority}')
            st.caption(advice)
            if detections:
                st.dataframe(detections, hide_index=True, width='stretch')
            from hashlib import sha256
            source_key = sha256(
                file.getvalue() + f"{model_path}:{conf}:{imgsz}".encode()
            ).hexdigest()
            if st.session_state.get('pi_upload_source_key') != source_key:
                st.session_state['pi_upload_source_key'] = source_key
                st.session_state['pi_upload_guidance'] = None

            if st.button('🤖 Explain with Llama (offline)', key='pi_llama_button'):
                with st.spinner('Generating locally with ExecuTorch…'):
                    try:
                        report = explain_defects(detections)
                    except Exception as err:
                        st.error(f'Llama explanation unavailable: {err}')
                    else:
                        explanation = report['llama_explanation'].strip()
                        if (len(explanation) < 60 or
                                not explanation.endswith(('.', '!', '?')) or
                                'use the following format' in explanation.lower()):
                            explanation = grounded_inspection_guidance(detections, report['summary'])
                            st.caption('Incomplete Llama response; YOLO-based inspection guidance shown.')
                        st.session_state['pi_upload_guidance'] = (
                            report['summary'], explanation
                        )
            upload_guidance = st.session_state.get('pi_upload_guidance')
            if upload_guidance:
                st.markdown('**YOLO summary**')
                st.write(upload_guidance[0])
                st.markdown('**Llama-assisted inspection guidance**')
                st.write(upload_guidance[1])
                st.caption('Advisory only. Human verification required.')
            ok, encoded = cv2.imencode('.png', annotated)
            if ok:
                png = encoded.tobytes()
                st.download_button('⬇️ Download annotated PNG', png, file_name='annotated_'+Path(file.name).stem+'.png', mime='image/png')
                if st.button('💾 Save inspection to SQLite'):
                    recommendation = upload_guidance[1] if upload_guidance else advice
                    id_ = save_inspection(
                        DB, OUT, filename=file.name, model=model_path,
                        conf=conf, imgsz=imgsz, detections=detections,
                        priority=priority, recommendation=recommendation,
                        annotated_png=png,
                    )
                    st.success(f'Inspection {id_} saved locally.')

# Camera streaming reads from the Raspberry Pi host, NOT from the remote browser.
# The Windows application camera code is unchanged by this feature.
def stop_pi_live_camera():
    camera = st.session_state.get('pi_live_capture')
    if camera is not None:
        camera.release()
    st.session_state['pi_live_capture'] = None
    st.session_state['pi_live_running'] = False


with live_tab:
    st.subheader('Live camera inspection')
    st.info(
        'Supported sources: USB/V4L2 webcams connected to the Pi, '
        'Raspberry Pi CSI Camera Modules (Picamera2 or rpicam-still), '
        'RTSP/HTTP IP cameras, and an optional live browser webcam on Windows.'
    )
    st.warning(
        'Continuous YOLO inference uses Raspberry Pi CPU power. '
        'Use a stable Pi 5 supply before running extended camera tests.'
    )
    if 'pi_live_running' not in st.session_state:
        st.session_state['pi_live_running'] = False
    if 'pi_live_capture' not in st.session_state:
        st.session_state['pi_live_capture'] = None

    is_running = st.session_state['pi_live_running']
    camera_type = st.selectbox(
        'Live video source',
        [
            'USB webcam on Raspberry Pi',
            'Raspberry Pi Camera Module (CSI)',
            'RTSP / HTTP network camera',
            'Browser webcam (Windows or phone) — WebRTC',
        ],
        disabled=is_running, key='pi_live_camera_type'
    )

    if camera_type.startswith('Browser webcam'):
        render_browser_live(
            model, conf, imgsz, show_annotations,
            DB, OUT, model_path, grounded_inspection_guidance
        )
    else:
        camera_index = 0
        if camera_type in ('USB webcam on Raspberry Pi', 'Raspberry Pi Camera Module (CSI)'):
            camera_index = st.number_input(
                'Camera index (0 = first camera, 1 = second camera)',
                min_value=0, max_value=10, value=0, step=1,
                disabled=is_running, key='pi_camera_index'
            )
            camera_source = int(camera_index)
            if camera_type == 'Raspberry Pi Camera Module (CSI)':
                st.caption(
                    'Uses Picamera2 if available; otherwise uses slow rpicam-still '
                    'captures. The CSI camera must be connected to the Pi and '
                    'recognized by the system camera software.'
                )
        else:
            camera_url = st.text_input(
                'RTSP or HTTP stream URL',
                placeholder='rtsp://192.168.1.20:554/stream',
                disabled=is_running, key='pi_camera_url'
            )
            camera_source = camera_url.strip()

        if st.button('🔎 Detect attached Pi cameras', key='pi_cam_inventory'):
            st.code(camera_inventory(), language='text')

        start_col, stop_col = st.columns(2)
        with start_col:
            if st.button('▶ Start live detection', key='pi_live_start', disabled=is_running):
                if camera_type == 'RTSP / HTTP network camera' and not str(camera_source).startswith(('rtsp://', 'http://', 'https://')):
                    st.error('Enter a valid rtsp://, http://, or https:// video stream URL.')
                else:
                    try:
                        camera = open_camera(
                            camera_type, index=int(camera_index), url=str(camera_source)
                        )
                    except CameraOpenError as error:
                        st.error(f'Camera unavailable: {error}')
                    else:
                        st.session_state['pi_live_capture'] = camera
                        st.session_state['pi_live_running'] = True
                        st.session_state['pi_live_last'] = None
                        st.session_state['pi_live_last_explanation'] = None
                        st.rerun()
    
        with stop_col:
            if st.button('■ Stop and review latest frame', key='pi_live_stop', disabled=not is_running):
                stop_pi_live_camera()
                st.rerun()
    
        if st.session_state['pi_live_running']:
            st.caption('Live frames are periodically annotated by YOLO. Updates are limited by Raspberry Pi CPU inference speed; this is not a high-FPS video pipeline.')
    
            @st.fragment(run_every='3s')
            def render_live_detection():
                camera = st.session_state.get('pi_live_capture')
                if camera is None or not camera.isOpened():
                    stop_pi_live_camera()
                    st.error('Camera disconnected.')
                    st.rerun()
                ok, frame = camera.read()
                if not ok or frame is None:
                    stop_pi_live_camera()
                    st.error('Could not read a camera frame. Check the video source.')
                    st.rerun()
    
                try:
                    annotated, detections, elapsed_ms = predict(frame, model, conf=conf, imgsz=imgsz)
                except Exception as error:
                    stop_pi_live_camera()
                    st.error(f'Live YOLO inference stopped: {error}')
                    st.rerun()
    
                priority, advice = review(detections)
                encoded_ok, encoded = cv2.imencode('.png', annotated)
                if encoded_ok:
                    st.session_state['pi_live_last'] = {
                        'annotated_png': encoded.tobytes(),
                        'original_rgb': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        'detections': detections,
                        'elapsed_ms': elapsed_ms,
                        'priority': priority,
                        'advice': advice,
                        'model': model_path,
                        'conf': conf,
                        'imgsz': imgsz,
                    }
                visible_frame = annotated if show_annotations else frame
                st.image(cv2.cvtColor(visible_frame, cv2.COLOR_BGR2RGB),
                         caption='Live YOLO annotations' if show_annotations else 'Live camera preview (annotations hidden)',
                         width='stretch')
                st.write(f'**Detected regions:** {len(detections)} | **YOLO inference:** {elapsed_ms:.0f} ms | **Review:** {priority}')
                st.caption(advice)
                if detections:
                    st.dataframe(detections, hide_index=True, width='stretch')
    
            render_live_detection()
        else:
            last = st.session_state.get('pi_live_last')
            if last is None:
                st.caption('Select a camera source and click Start live detection. No camera is needed for image uploads.')
            else:
                st.markdown('#### Latest captured inspection')
                saved_frame = cv2.imdecode(np.frombuffer(last['annotated_png'], dtype=np.uint8), cv2.IMREAD_COLOR)
                if show_annotations:
                    st.image(cv2.cvtColor(saved_frame, cv2.COLOR_BGR2RGB),
                             caption='Final annotated live frame', width='stretch')
                else:
                    st.image(last['original_rgb'], caption='Final frame (annotations hidden)', width='stretch')
                st.write(f"**Detected regions:** {len(last['detections'])} | **YOLO inference:** {last['elapsed_ms']:.0f} ms | **Review:** {last['priority']}")
                st.caption(last['advice'])
                if last['detections']:
                    st.dataframe(last['detections'], hide_index=True, width='stretch')
                st.download_button('⬇️ Download live-frame annotations', last['annotated_png'],
                                   file_name='live_inspection_annotated.png', mime='image/png',
                                   key='pi_live_download')
    
                if st.button('🤖 Explain captured live frame with Llama (offline)', key='pi_live_llama'):
                    with st.spinner('Generating locally with ExecuTorch…'):
                        try:
                            report = explain_defects(last['detections'])
                            explanation = report['llama_explanation'].strip()
                            if (len(explanation) < 60 or
                                    not explanation.endswith(('.', '!', '?')) or
                                    'use the following format' in explanation.lower()):
                                explanation = grounded_inspection_guidance(last['detections'], report['summary'])
                            st.session_state['pi_live_last_explanation'] = (
                                report['summary'], explanation
                            )
                        except Exception as error:
                            st.error(f'Llama explanation unavailable: {error}')
                existing_explanation = st.session_state.get('pi_live_last_explanation')
                if existing_explanation:
                    st.markdown('**YOLO summary:**')
                    st.write(existing_explanation[0])
                    st.markdown('**Llama-assisted inspection guidance:**')
                    st.write(existing_explanation[1])
                    st.caption('Advisory only. Human inspection required.')
    
                if st.button('💾 Save captured live frame to SQLite', key='pi_live_save'):
                    guidance = st.session_state.get('pi_live_last_explanation')
                    inspection_id = save_inspection(
                        DB, OUT, filename='live_camera_capture.png',
                        model=last['model'], conf=last['conf'], imgsz=last['imgsz'],
                        detections=last['detections'], priority=last['priority'],
                        recommendation=guidance[1] if guidance else last['advice'],
                        annotated_png=last['annotated_png'],
                    )
                    st.success(f'Live inspection {inspection_id} saved. Open Inspection history to review it.')

with history_tab:
    rows = load_inspections(DB)
    if not rows:
        st.info('No saved inspections yet.')
    else:
        cols = ['timestamp_utc','filename','detection_count','priority','recommendation']
        st.dataframe([{k:r[k] for k in cols} for r in rows], width='stretch', hide_index=True)
        st.download_button(
            '⬇️ Export CSV', export_inspections_csv(rows),
            file_name='pi_inspections.csv', mime='text/csv'
        )
        idx = st.selectbox('View saved image', list(range(len(rows))),
                           format_func=lambda i: f"{rows[i]['timestamp_utc']} | {rows[i]['filename']}")
        saved = Path(rows[idx]['annotated_file'])
        if saved.is_file():
            st.image(str(saved), width='stretch')
        st.json(json.loads(rows[idx]['detections_json']))
st.divider()
st.warning('Research prototype only. NEU-DET lacks verified normal examples. No detections does not prove quality acceptance.')
