#!/usr/bin/env bash
# Run on Pi as pi, not as root.
set -euo pipefail
[ "$(id -un)" = pi ] || { echo 'Run as the pi user.'; exit 1; }
[ "$(uname -m)" = aarch64 ] || { echo 'Raspberry Pi ARM64 required.'; exit 1; }
ROOT="$HOME/Industrial_Defect_Detection"
PY="$HOME/soc-venvs/cpu-validation-BdkS2KBp/.venv/bin/python"
[ -x "$PY" ]
[ -f "$ROOT/src/learning/model_registry.py" ]
[ -f "$ROOT/src/dashboard/model_history.py" ]
systemctl cat steel-inspector.service >/dev/null
systemctl cat steel-learning.timer >/dev/null
sudo -v
TMP=$(mktemp -d)
BACKUP="$HOME/soc-project-backups/operator_addons_$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p "$BACKUP"
TIMER_STOPPED=0
APP_STOPPED=0
NEW_HELPER=0
NEW_RULE=0
COMMITTED=0
cleanup() {
    code=$?
    if [ "$COMMITTED" = 0 ]; then
        if [ -f "$BACKUP/manifest.json" ]; then
            sudo systemctl stop steel-inspector.service || true
            "$PY" - "$ROOT" "$BACKUP" <<'PYRESTORE' || true
import json,sys
from pathlib import Path
root, backup=map(Path,sys.argv[1:])
for name, existed in json.loads((backup/'manifest.json').read_text()).items():
    p=root/name
    if existed: p.write_bytes((backup/name).read_bytes())
    else: p.unlink(missing_ok=True)
print('Original application sources restored.')
PYRESTORE
            APP_STOPPED=1
        fi
        if [ "$NEW_RULE" = 1 ]; then sudo rm -f /etc/sudoers.d/steel-operator-power || true; fi
        if [ "$NEW_HELPER" = 1 ]; then sudo rm -f /usr/local/sbin/steel-operator-power || true; fi
    fi
    if [ "$APP_STOPPED" = 1 ]; then sudo systemctl start steel-inspector.service || true; fi
    if [ "$TIMER_STOPPED" = 1 ]; then sudo systemctl start steel-learning.timer || true; fi
    rm -rf "$TMP"
    exit "$code"
}
trap cleanup EXIT

cat > "$TMP/operator_backend.py" <<'STEEL_ADDON_SOURCE_0'
"""Local, advisory defect notifications and curated multilingual checks."""
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

WORK = Path.home() / 'soc-retraining'
ALERT_DB = WORK / 'operator_alerts.sqlite3'
COOLDOWN = 60
# Advisory visual checks, not machine-generated translations or certified actions.
GUIDANCE = {
 'crazing': ('Crazing', 'Inspect the marked area for a network of fine cracks.', 'நுண் விரிசல்கள்', 'குறியிட்ட பகுதியில் நுண்ணிய விரிசல்கள் வலைபோல் உள்ளதா எனப் பாருங்கள்.', 'सूक्ष्म दरारें', 'चिह्नित क्षेत्र में महीन दरारों का जाल देखें।'),
 'inclusion': ('Inclusion', 'Inspect the marked area for embedded foreign material.', 'அந்நியப் பொருள் உட்சேர்வு', 'குறியிட்ட பகுதியில் அந்நியப் பொருள் பதிந்துள்ளதா எனப் பாருங்கள்.', 'बाहरी पदार्थ का समावेशन', 'चिह्नित क्षेत्र में धँसे हुए बाहरी पदार्थ की जाँच करें।'),
 'patches': ('Patches', 'Compare the marked patch with the surrounding surface under consistent lighting.', 'திட்டுகள்', 'ஒரே மாதிரியான வெளிச்சத்தில் குறியிட்ட திட்டைச் சுற்றியுள்ள மேற்பரப்புடன் ஒப்பிடுங்கள்.', 'धब्बे', 'समान रोशनी में चिह्नित धब्बे की आसपास की सतह से तुलना करें।'),
 'pitted_surface': ('Pitted surface', 'Inspect the marked area for small depressions; verify with an appropriate inspection method.', 'குழிவுள்ள மேற்பரப்பு', 'குறியிட்ட பகுதியில் சிறு குழிகள் உள்ளதா எனப் பார்த்து பொருத்தமான ஆய்வு முறையில் உறுதிசெய்யுங்கள்.', 'गड्ढेदार सतह', 'चिह्नित क्षेत्र में छोटे गड्ढे देखें और उपयुक्त निरीक्षण विधि से पुष्टि करें।'),
 'rolled_in_scale': ('Rolled-in scale', 'Inspect the marked area for embedded scale or irregular surface texture.', 'உருட்டலில் பதிந்த ஆக்சைடு படலம்', 'குறியிட்ட பகுதியில் பதிந்த ஆக்சைடு படலம் அல்லது சீரற்ற மேற்பரப்பு உள்ளதா எனப் பாருங்கள்.', 'लुढ़काई में धँसी ऑक्साइड परत', 'चिह्नित क्षेत्र में धँसी ऑक्साइड परत या असमान बनावट की जाँच करें।'),
 'scratches': ('Scratches', 'Inspect the marked area and contact surfaces.', 'கீறல்கள்', 'குறியிட்ட பகுதியையும் தொடர்பு கொள்ளும் மேற்பரப்புகளையும் ஆய்வு செய்யுங்கள்.', 'खरोंचें', 'चिह्नित क्षेत्र और संपर्क सतहों की जाँच करें।'),
}
NOTICE = {
 'English': ('Local QC visual checks', 'Human verification required. These are draft checks, not confirmed causes or acceptance criteria.', 'No target regions detected. This does not establish that the surface is defect-free.', 'Unrecognized defect class; request human review.'),
 'தமிழ்': ('உள்ளூர் தர ஆய்வு வழிகாட்டுதல்', 'மனித ஆய்வு அவசியம். இவை வரைவு வழிகாட்டுதல்கள்; உறுதிசெய்யப்பட்ட காரணங்களோ ஏற்றுக்கொள்ளும் அளவுகோல்களோ அல்ல.', 'இலக்கு குறைபாடு கண்டறியப்படவில்லை. இதனால் மேற்பரப்பு குறைபாடற்றது என உறுதி செய்ய முடியாது.', 'அறியப்படாத குறைபாட்டு வகை; மனித ஆய்வு தேவை.'),
 'हिन्दी': ('स्थानीय गुणवत्ता निरीक्षण मार्गदर्शन', 'मानव सत्यापन आवश्यक है। ये प्रारूप जाँचें हैं, प्रमाणित कारण या स्वीकृति मानदंड नहीं।', 'लक्षित दोष नहीं मिला। इससे सतह का दोष-मुक्त होना सिद्ध नहीं होता।', 'अपरिचित दोष वर्ग; मानव निरीक्षण आवश्यक है।'),
}

def normalize(value):
    return str(value).strip().lower().replace('-', '_').replace(' ', '_')

def local_guidance(detections, language):
    language = language if language in NOTICE else 'English'
    offset = {'English': 0, 'தமிழ்': 2, 'हिन्दी': 4}[language]
    heading, notice, empty, unknown = NOTICE[language]
    names = sorted({normalize(d.get('class', '')) for d in detections})
    checks = [(GUIDANCE[n][offset], GUIDANCE[n][offset+1]) if n in GUIDANCE else (n, unknown) for n in names]
    return heading, notice, checks, empty

def connect():
    WORK.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(ALERT_DB, timeout=5)
    c.row_factory = sqlite3.Row
    return c

def initialize():
    with closing(connect()) as c, c:
        c.execute('''CREATE TABLE IF NOT EXISTS alerts (
          id INTEGER PRIMARY KEY, created_utc TEXT NOT NULL,
          epoch REAL NOT NULL, signature TEXT NOT NULL,
          priority TEXT NOT NULL, classes TEXT NOT NULL,
          regions INTEGER NOT NULL, acknowledged_utc TEXT)''')
        c.execute('CREATE INDEX IF NOT EXISTS alerts_signature_epoch ON alerts(signature, epoch)')

def record_alert(detections, priority):
    if str(priority).upper() not in ('HIGH REVIEW', 'MEDIUM REVIEW') or not detections:
        return False
    initialize()
    classes = sorted({normalize(d.get('class', 'unknown')) for d in detections})
    priority = str(priority).upper()
    signature = hashlib.sha256(json.dumps([priority, classes]).encode()).hexdigest()
    now = time.time()
    with closing(connect()) as c, c:
        c.execute('BEGIN IMMEDIATE')
        if c.execute('SELECT 1 FROM alerts WHERE signature=? AND epoch>? LIMIT 1', (signature, now-COOLDOWN)).fetchone():
            return False
        c.execute('INSERT INTO alerts(created_utc,epoch,signature,priority,classes,regions) VALUES(?,?,?,?,?,?)',
                  (datetime.now(timezone.utc).isoformat(), now, signature, priority, ', '.join(classes), len(detections)))
    return True

def alerts(limit=200):
    initialize()
    with closing(connect()) as c:
        return [dict(r) for r in c.execute('SELECT id,created_utc,priority,classes,regions,acknowledged_utc FROM alerts ORDER BY id DESC LIMIT ?', (int(limit),))]

def unread():
    initialize()
    with closing(connect()) as c:
        return c.execute('SELECT COUNT(*) FROM alerts WHERE acknowledged_utc IS NULL').fetchone()[0]

def acknowledge(alert_id):
    initialize()
    with closing(connect()) as c, c:
        c.execute('UPDATE alerts SET acknowledged_utc=? WHERE id=? AND acknowledged_utc IS NULL', (datetime.now(timezone.utc).isoformat(), int(alert_id)))
STEEL_ADDON_SOURCE_0

cat > "$TMP/operator_ui.py" <<'STEEL_ADDON_SOURCE_1'
"""Pi-only batch inspection, alert inbox, multilingual QC, and recovery controls."""
import csv
import hashlib
import io
import logging
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from learning import operator_backend as ops
from learning import model_registry as reg
from learning.feedback_store import collect_image
from common import review
from inspection_store import save_inspection

MAX_FILES = 20
MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 12_000_000


def render_local_guidance(detections):
    language = st.session_state.get('steel_qc_language', 'English')
    heading, notice, checks, empty = ops.local_guidance(detections, language)
    with st.expander(heading, expanded=language != 'English'):
        for title, check in checks:
            st.markdown('**' + title + '**')
            st.write(check)
        if not checks:
            st.write(empty)
        st.caption(notice)
        st.caption('Offline curated checks; the original detailed Llama report remains in English.')


@st.fragment(run_every='10s')
def render_alert_counter():
    try:
        count = ops.unread()
        if count:
            st.warning(f'{count} unacknowledged QC alert(s) · open Alerts')
        else:
            st.caption('No unacknowledged QC alerts')
    except Exception as exc:
        st.caption('Alert store unavailable: ' + str(exc))


def render_alerts():
    st.subheader('QC alerts')
    st.caption('Local advisory alerts for medium/high review. Repeated class/priority combinations are grouped for 60 seconds; this is not a safety alarm or a production stop.')
    try:
        rows = ops.alerts()
        count = ops.unread()
    except Exception as exc:
        st.error(str(exc)); return
    a,b = st.columns(2)
    a.metric('Unacknowledged alerts', count, border=True)
    b.metric('Recent records', len(rows), border=True)
    if not rows:
        st.info('No alerts recorded yet.'); return
    for row in rows[:30]:
        with st.container(border=True):
            a,b = st.columns([4,1])
            with a:
                st.markdown(f"**#{row['id']} · {row['priority']}**")
                st.write(f"{row['classes']} · {row['regions']} region(s)")
                st.caption(row['created_utc'])
            with b:
                if row['acknowledged_utc']:
                    st.caption('Acknowledged')
                elif st.button('Acknowledge', key=f"steel_alert_ack_{row['id']}"):
                    ops.acknowledge(row['id']); st.rerun()
    with st.expander('Recent alert history'):
        st.dataframe(rows, hide_index=True, width='stretch')


def decode_upload(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError('Image exceeds 12 MiB limit')
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in ('JPEG','PNG','BMP'):
            raise ValueError('Unsupported image format')
        w,h = image.size
        if w*h > MAX_PIXELS:
            raise ValueError('Image exceeds 12 megapixel limit')
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or not frame.size:
        raise ValueError('Cannot decode image')
    return frame


def render_batch(model, conf, imgsz, db, out, model_path):
    # Import lazily to avoid dashboard_ui importing this module circularly.
    from dashboard_ui import predict_for_dashboard
    st.subheader('Batch inspection')
    st.caption('Sequential local inference, up to 20 images per batch. Successful inspections are saved to local history. Training samples remain pending review.')
    files = st.file_uploader('Choose steel surface images', type=['jpg','jpeg','png','bmp'], accept_multiple_files=True, key='steel_batch_files')
    if st.button('Inspect and save batch', type='primary', key='steel_batch_start', disabled=not files):
        if len(files) > MAX_FILES:
            st.error(f'Maximum {MAX_FILES} images per batch.'); return
        if st.session_state.get('pi_live_running', False):
            st.warning('Stop live inspection before starting a batch.'); return
        # Avoid starting while a scheduled training job is active/activating.
        state = service_state('steel-learning.service')
        if state not in ('inactive','failed'):
            st.warning('Training service is busy or unavailable; try after it finishes.'); return
        rows = []
        seen = set()
        progress = st.progress(0)
        # Hold the activity lock for the entire batch, not just individual predictions.
        @reg.inspection_activity
        def run_batch():
            for i, file in enumerate(files):
                row = {'File': Path(file.name).name, 'Regions': None, 'Review': '', 'Inference ms': None, 'Inspection ID': '', 'Status': ''}
                try:
                    raw = file.getvalue()
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest in seen:
                        row['Status'] = 'Skipped duplicate in batch'
                    else:
                        seen.add(digest)
                        frame = decode_upload(raw)
                        annotated, detections, ms = predict_for_dashboard(frame, model, conf=conf, imgsz=imgsz)
                        priority, advice = review(detections)
                        ok, encoded = cv2.imencode('.png', annotated)
                        if not ok: raise RuntimeError('Cannot encode result')
                        ident = save_inspection(db, out, filename=file.name, model=model_path, conf=conf, imgsz=imgsz, detections=detections, priority=priority, recommendation=advice, annotated_png=encoded.tobytes())
                        row.update({'Regions':len(detections),'Review':priority,'Inference ms':round(ms,1),'Inspection ID':ident,'Status':'Saved'})
                        try:
                            collect_image(frame, detections, file.name)
                        except Exception as exc:
                            row['Status'] = 'Saved; learning collection failed: ' + str(exc)
                except Exception as exc:
                    row['Status'] = 'Failed: ' + str(exc)
                rows.append(row)
                progress.progress((i+1)/len(files))
        run_batch()
        st.session_state['steel_batch_results'] = rows
    rows = st.session_state.get('steel_batch_results', [])
    if rows:
        st.dataframe(rows, hide_index=True, width='stretch')
        st.caption('Annotated images can be opened from Inspection history. Batch mode uses the deterministic QC review, not per-image Llama generation.')
        out_csv = io.StringIO()
        writer = csv.DictWriter(out_csv, fieldnames=list(rows[0]))
        writer.writeheader()
        # Prevent spreadsheet formula injection from filenames/errors.
        writer.writerows({k: ("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@','\t','\r')) else v) for k,v in row.items()} for row in rows)
        st.download_button('Download batch CSV', out_csv.getvalue(), 'batch_inspection.csv', 'text/csv', key='steel_batch_csv')


def service_state(unit):
    try:
        result = subprocess.run(['/usr/bin/systemctl','show','-p','ActiveState','--value',unit], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except (OSError, subprocess.TimeoutExpired):
        return 'unknown'


def db_check(path):
    if not path.is_file(): return 'Not present'
    try:
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=3)) as c:
            results = [r[0] for r in c.execute('PRAGMA quick_check')]
        return 'OK' if results == ['ok'] else '; '.join(results)[:300]
    except Exception as exc:
        return 'Unavailable: ' + str(exc)


def render_recovery(db):
    st.subheader('Power and recovery')
    st.caption('Normal shutdown and reboot allow Linux to stop services and flush storage. Sudden power removal can still cause data loss.')
    units = ['steel-inspector.service','steel-learning.service','steel-learning.timer']
    st.dataframe([{'Service':unit,'State':service_state(unit)} for unit in units], hide_index=True,width='stretch')
    if st.button('Check local storage integrity', key='steel_db_check'):
        for name,path in [('Inspections',Path(db)),('Learning',reg.WORK/'feedback.sqlite3'),('Alerts',ops.ALERT_DB)]:
            st.write(f'{name}: {db_check(path)}')
    if st.button('Reload dashboard', key='steel_recovery_reload'):
        st.rerun()
    st.divider()
    st.warning('Stop camera streams and finish saving inspections first. These controls affect the entire Raspberry Pi; anyone with access to this dashboard can request them.')
    with st.form('steel_power_confirm'):
        action = st.selectbox('Power action', ['Shutdown','Reboot'])
        confirm = st.checkbox('I confirm that all inspection work is saved and camera streams are stopped.')
        submitted = st.form_submit_button('Execute power action', type='primary')
    if submitted:
        if not confirm:
            st.warning('Confirmation is required.'); return
        if st.session_state.get('pi_live_running',False) or not reg.idle_for(5):
            st.warning('Inspection is active or recently used. Stop inspection first.'); return
        if service_state('steel-learning.service') not in ('inactive','failed'):
            st.warning('Training is active or its state is unknown. Power action refused.'); return
        try:
            result = subprocess.run(['/usr/bin/sudo','-n','/usr/local/sbin/steel-operator-power','poweroff' if action=='Shutdown' else 'reboot'], capture_output=True,text=True,timeout=15)
            if result.returncode: st.error((result.stderr or result.stdout or 'Power action refused').strip())
            else: st.info('Power action accepted by the operating system.')
        except (OSError, subprocess.TimeoutExpired) as exc:
            st.error(f'Power action result unavailable: {exc}')
STEEL_ADDON_SOURCE_1

cat > "$TMP/patch_operator.py" <<'STEEL_ADDON_SOURCE_2'
import ast
import json
import sys
from pathlib import Path


def patch_app(source):
    if 'from operator_ui import render_batch' in source:
        return source
    tree = ast.parse(source)
    tabs = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr=='tabs']
    if len(tabs)!=1: raise RuntimeError('Expected one top-level tabs assignment')
    t = tabs[0]
    if len(t.targets)!=1 or not isinstance(t.targets[0],ast.Tuple) or not isinstance(t.value.args[0],(ast.List,ast.Tuple)):
        raise RuntimeError('Unexpected tabs structure')
    labels = [ast.literal_eval(x) for x in t.value.args[0].elts]
    targets = [x.id for x in t.targets[0].elts]
    if len(labels)!=4 or 'Model history' not in labels or 'models_tab' not in targets:
        raise RuntimeError('Automatic learning dashboard must be installed first')
    replacement = '''from operator_ui import render_batch, render_alerts, render_recovery, render_alert_counter
st.sidebar.selectbox('QC guidance language', ['English', 'தமிழ்', 'हिन्दी'], key='steel_qc_language')
with st.sidebar:
    render_alert_counter()
''' + ', '.join(targets+['batch_tab','alerts_tab','power_tab']) + ' = st.tabs(' + repr(labels+['Batch inspection','Alerts','Power & recovery']) + ')\n'
    lines = source.splitlines(keepends=True)
    lines[t.lineno-1:t.end_lineno] = [replacement]
    return ''.join(lines) + '''\n\n# Local operator additions\nwith batch_tab:\n    render_batch(model, conf, imgsz, DB, OUT, model_path)\nwith alerts_tab:\n    render_alerts()\nwith power_tab:\n    render_recovery(DB)\n'''


def patch_helper(source):
    if '# Steel operator additions' in source:
        return source
    names = {n.name for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    if not {'predict_for_dashboard','detection_details'} <= names:
        raise RuntimeError('Unexpected dashboard helper structure')
    return source + '''\n\n# Steel operator additions\nimport logging as _operator_logging\nfrom learning.operator_backend import record_alert as _record_operator_alert\nfrom common import review as _operator_review\nfrom operator_ui import render_local_guidance as _render_operator_guidance\n_previous_dashboard_predict = predict_for_dashboard\n_previous_detection_details = detection_details\n\ndef predict_for_dashboard(frame, model, **kwargs):\n    annotated, detections, elapsed = _previous_dashboard_predict(frame, model, **kwargs)\n    try:\n        priority, _ = _operator_review(detections)\n        _record_operator_alert(detections, priority)\n    except Exception:\n        _operator_logging.getLogger(__name__).exception('Optional QC alert recording failed')\n    return annotated, detections, elapsed\n\ndef detection_details(detections):\n    _previous_detection_details(detections)\n    _render_operator_guidance(detections)\n'''


def main():
    root, package, backup = map(Path,sys.argv[1:4])
    app=root/'src/dashboard/app_pi.py'
    helper=root/'src/dashboard/dashboard_ui.py'
    files={app:patch_app(app.read_text()),helper:patch_helper(helper.read_text()),root/'src/dashboard/operator_ui.py':(package/'operator_ui.py').read_text(),root/'src/learning/operator_backend.py':(package/'operator_backend.py').read_text()}
    for path,source in files.items(): compile(source,str(path),'exec')
    manifest={}
    for path in files:
        rel=str(path.relative_to(root)); manifest[rel]=path.exists()
        if path.exists():
            target=backup/rel; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(path.read_bytes())
    (backup/'manifest.json').write_text(json.dumps(manifest))
    for path,source in files.items():
        path.write_text(source,encoding='utf-8'); print('Installed:',path.relative_to(root))
    sys.path.insert(0,str(root/'src'))
    from learning.operator_backend import initialize, unread
    initialize(); print('Alert database ready; unacknowledged:',unread())

if __name__=='__main__': main()
STEEL_ADDON_SOURCE_2

cat > "$TMP/power_helper" <<'STEEL_ADDON_SOURCE_3'
#!/bin/sh
# Root-owned helper. Only these two exact operations are authorized.
set -eu
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
[ "$#" -eq 1 ] || exit 2
case "$1" in poweroff|reboot) action="$1" ;; *) exit 2 ;; esac
exec 9>/run/lock/steel-operator-power.lock
flock -n 9 || { echo 'Power request already running.' >&2; exit 1; }
timer_was_active=0
if systemctl is-active --quiet steel-learning.timer; then timer_was_active=1; fi
restore_timer() { if [ "$timer_was_active" -eq 1 ]; then systemctl start steel-learning.timer || true; fi; }
# Do not allow the timer to start another training job during shutdown.
if ! systemctl stop steel-learning.timer; then restore_timer; exit 1; fi
state=$(systemctl show --property=ActiveState --value steel-learning.service) || { restore_timer; exit 1; }
case "$state" in inactive|failed) ;; *) restore_timer; echo 'Training service is busy; power action refused.' >&2; exit 1 ;; esac
# Normal systemd shutdown handles application stop and disk flush.
if ! systemctl --no-block "$action"; then restore_timer; exit 1; fi
STEEL_ADDON_SOURCE_3

"$PY" -m py_compile "$TMP"/*.py
sh -n "$TMP/power_helper"
"$PY" -c 'import streamlit, cv2, numpy, PIL, sqlite3'
printf '%s\n' 'pi ALL=(root) NOPASSWD: /usr/local/sbin/steel-operator-power poweroff, /usr/local/sbin/steel-operator-power reboot' > "$TMP/power_sudoers"
sudo visudo -cf "$TMP/power_sudoers"
# Refuse to overwrite unrelated privileged files.
if sudo test -e /usr/local/sbin/steel-operator-power; then sudo cmp -s "$TMP/power_helper" /usr/local/sbin/steel-operator-power || { echo 'Different power helper already exists; aborting.'; exit 1; }; fi
if sudo test -e /etc/sudoers.d/steel-operator-power; then sudo cmp -s "$TMP/power_sudoers" /etc/sudoers.d/steel-operator-power || { echo 'Different power policy already exists; aborting.'; exit 1; }; fi
sudo systemctl stop steel-learning.timer
TIMER_STOPPED=1
state=$(systemctl show -p ActiveState --value steel-learning.service)
case "$state" in inactive|failed) ;; *) echo 'Training service active; no application files changed.'; exit 1;; esac
sudo systemctl stop steel-inspector.service
APP_STOPPED=1
"$PY" "$TMP/patch_operator.py" "$ROOT" "$TMP" "$BACKUP"
if ! sudo test -e /usr/local/sbin/steel-operator-power; then
    NEW_HELPER=1
    sudo install -o root -g root -m 755 "$TMP/power_helper" /usr/local/sbin/steel-operator-power
fi
if ! sudo test -e /etc/sudoers.d/steel-operator-power; then
    NEW_RULE=1
    sudo install -o root -g root -m 440 "$TMP/power_sudoers" /etc/sudoers.d/steel-operator-power
fi
sudo visudo -c
sudo systemctl start steel-inspector.service
ready=0
for i in $(seq 1 30); do
    if curl -fsS --max-time 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
done
[ "$ready" = 1 ] || { echo 'Dashboard health endpoint failed; restoring sources.'; exit 1; }
sudo systemctl start steel-learning.timer
TIMER_STOPPED=0
APP_STOPPED=0
COMMITTED=1
printf '\n=== OPERATOR FEATURES INSTALLED ===\n'
echo '4: Batch inspection (20 images per run)'
echo '9: Persistent local QC alerts and acknowledgement'
echo '10: English / Tamil / Hindi offline visual checks'
echo '12: Confirmed shutdown/reboot and database recovery diagnostics'
echo "Source backup: $BACKUP"
systemctl is-active steel-inspector.service
systemctl list-timers steel-learning.timer --all --no-pager
cat "$HOME/soc-retraining/training_status.json"
echo 'Open or refresh the local dashboard. Power controls have not been executed.'
