#!/usr/bin/env bash
# Full source installer: run as the Pi desktop user, not root.
set -euo pipefail
if [ "$(id -u)" -eq 0 ]; then echo 'Run as pi, not sudo bash'; exit 1; fi
PROJECT="$HOME/Industrial_Defect_Detection"
WORK="$HOME/soc-retraining"
CPU_ENV="$HOME/soc-venvs/cpu-validation-BdkS2KBp/.venv"
CPU_PYTHON="$CPU_ENV/bin/python"
[ "$(uname -m)" = aarch64 ] || { echo 'This installer targets Raspberry Pi ARM64'; exit 1; }
test -x "$CPU_PYTHON"
test -f "$PROJECT/src/learning/feedback_store.py"
test -f "$PROJECT/src/dashboard/learning_feedback.py"
systemctl cat steel-inspector.service >/dev/null
if systemctl is-active --quiet steel-learning.service; then echo 'A training job is running. Do not install during a job.'; exit 1; fi
sudo -v
if systemctl cat steel-learning.timer >/dev/null 2>&1; then sudo systemctl stop steel-learning.timer; fi
TMP="$(mktemp -d)"
APP_STOPPED=0
cleanup() {
    rm -rf -- "$TMP"
    if [ "$APP_STOPPED" = 1 ]; then sudo systemctl start steel-inspector.service || true; fi
}
trap cleanup EXIT

cat > "$TMP/model_registry.py" <<'STEEL_PYTHON_0'
"""Local immutable model versions and atomic activation state."""
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path

WORK = Path.home() / 'soc-retraining'
REG = WORK / 'model_registry'
STATE = REG / 'registry.json'
STATUS = WORK / 'training_status.json'
CONFIG = WORK / 'config' / 'learning.json'
ACTIVITY = WORK / 'inspection_activity'
LOCK = REG / 'registry.lock'
BASE_ID = 'v000-baseline'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def exclusive():
    REG.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Training or another model operation is running.') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(folder):
    folder = Path(folder)
    if not folder.is_dir() or not (folder / 'metadata.yaml').is_file():
        raise ValueError(f'Invalid NCNN folder: {folder}')
    if not list(folder.glob('*.param')) or not list(folder.glob('*.bin')):
        raise ValueError(f'Missing NCNN parameter or weight files: {folder}')
    files = [p for p in folder.rglob('*') if p.is_file()]
    if any(p.is_symlink() for p in folder.rglob('*')):
        raise ValueError('Symlinks inside model artifacts are not allowed')
    return {str(p.relative_to(folder)): digest(p) for p in sorted(files)}


def read_state():
    state = read_json(STATE)
    if state['active'] not in state['versions']:
        raise RuntimeError('Registry active version missing')
    return state


def entry_path(entry):
    path = REG / entry['id'] / 'detector_ncnn_model'
    if not path.is_dir():
        raise RuntimeError(f'Registered artifact missing: {path}')
    return path


_verified_versions = set()


def active_path():
    state = read_state()
    entry = state['versions'][state['active']]
    if entry['id'] not in _verified_versions:
        try:
            verify_entry(entry)
        except (OSError, ValueError, RuntimeError):
            return recover_active(entry['id'])
        _verified_versions.add(entry['id'])
    return entry_path(entry)


def recover_active(failed_id):
    with exclusive():
        state = read_state()
        if state['active'] != failed_id:
            entry = state['versions'][state['active']]
            verify_entry(entry)
            return entry_path(entry)
        cursor = state['versions'][failed_id].get('parent')
        while cursor:
            entry = state['versions'][cursor]
            try:
                verify_entry(entry)
            except (OSError, ValueError, RuntimeError):
                cursor = entry.get('parent')
                continue
            if entry['ever_active']:
                state['active'] = cursor
                state['events'].append({'time':now(),'action':'integrity_rollback','from':failed_id,'to':cursor})
                write_json(STATE,state)
                _verified_versions.add(cursor)
                return entry_path(entry)
            cursor=entry.get('parent')
        raise RuntimeError('No intact previous active model found; original export preserved')


def verify_entry(entry):
    if artifact_hashes(entry_path(entry)) != entry['hashes']:
        raise RuntimeError('NCNN artifact integrity check failed')
    checkpoint = REG / entry['id'] / 'best.pt'
    if entry.get('pt_sha') and (not checkpoint.is_file() or digest(checkpoint) != entry['pt_sha']):
        raise RuntimeError('Trainable checkpoint integrity check failed')


def initialize(original):
    with exclusive():
        if STATE.exists():
            state = read_state()
            verify_entry(state['versions'][state['active']])
            return state
        folder = REG / BASE_ID
        if folder.exists():
            raise RuntimeError(f'Incomplete baseline registry exists: {folder}; inspect before retry')
        temp = Path(tempfile.mkdtemp(prefix='.baseline-', dir=REG))
        try:
            target = temp / 'detector_ncnn_model'
            shutil.copytree(original, target)
            hashes = artifact_hashes(target)
            if hashes != artifact_hashes(original):
                raise RuntimeError('Baseline copy hash mismatch')
            temp.rename(folder)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
        entry = {'id': BASE_ID, 'created_at': now(), 'parent': None,
                 'ever_active': True, 'decision': 'baseline', 'hashes': hashes,
                 'pt_sha': None, 'metrics': None}
        state = {'active': BASE_ID, 'versions': {BASE_ID: entry},
                 'consumed': [], 'evaluation_fingerprint': None,
                 'events': [{'time': now(), 'action': 'initialize', 'to': BASE_ID}]}
        write_json(STATE, state)
        return state


def mark_activity():
    WORK.mkdir(parents=True, exist_ok=True)
    ACTIVITY.touch()


def inspection_activity(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        WORK.mkdir(parents=True, exist_ok=True)
        with ACTIVITY.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            try:
                mark_activity()
                return function(*args, **kwargs)
            finally:
                mark_activity()
                fcntl.flock(stream, fcntl.LOCK_UN)
    return wrapped


def idle_for(seconds):
    try:
        with ACTIVITY.open('r') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return time.time() - os.fstat(stream.fileno()).st_mtime >= seconds
    except FileNotFoundError:
        return True


def rollback(target):
    with exclusive():
        state = read_state()
        old = state['active']
        if target == old:
            return
        entry = state['versions'].get(target)
        if not entry or not entry['ever_active']:
            raise ValueError('Only previously activated versions can be restored')
        if not idle_for(15):
            raise RuntimeError('Stop live inspection and wait 15 seconds before rollback')
        verify_entry(entry)
        state['active'] = target
        state['events'].append({'time': now(), 'action': 'rollback', 'from': old, 'to': target})
        write_json(STATE, state)
STEEL_PYTHON_0

cat > "$TMP/dataset_builder.py" <<'STEEL_PYTHON_1'
"""Build immutable train/validation snapshots from reviewed local images."""
import hashlib
import json
import math
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
import cv2
import numpy as np
import yaml
from .feedback_store import DB, CLASSES, image_fingerprint, normalize_class, initialize

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def pictures(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXT)


def parse_names(value):
    names = [value[i] if i in value else value[str(i)] for i in range(len(value))] if isinstance(value, dict) else list(value)
    if [normalize_class(n) for n in names] != list(CLASSES):
        raise ValueError('Dataset class IDs do not match feedback class order')
    return names


def check_label(path):
    if not path.is_file():
        raise ValueError(f'Missing YOLO annotation: {path}')
    classes = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'Invalid detection annotation: {path}')
        c = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        if not (0 <= c < len(CLASSES) and all(math.isfinite(v) for v in (x,y,w,h))
                and 0 < w <= 1 and 0 < h <= 1
                and x-w/2 >= -0.0001 and y-h/2 >= -0.0001
                and x+w/2 <= 1.0001 and y+h/2 <= 1.0001):
            raise ValueError(f'Invalid YOLO box: {path}')
        classes.add(c)
    return classes


def base_rows(root, split):
    images = root / split / 'images'
    labels = root / split / 'labels'
    for image in pictures(images):
        label = labels / image.relative_to(images).with_suffix('.txt')
        classes = check_label(label)
        frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f'Unreadable dataset image: {image}')
        yield image, label, image_fingerprint(frame), classes


def approved_count():
    initialize()
    with closing(sqlite3.connect(DB)) as conn:
        return conn.execute("SELECT COUNT(*) FROM samples WHERE status='approved'").fetchone()[0]


def build(root, dest, consumed, threshold, expected_fingerprint):
    root, dest = Path(root), Path(dest)
    names = parse_names(yaml.safe_load((root / 'data.yaml').read_text())['names'])
    seen = {}
    rows = {}
    val_classes = set()
    for split in ('val', 'test', 'train'):
        rows[split] = list(base_rows(root, split))
        if not rows[split]:
            raise ValueError(f'Original {split} dataset is empty: {root / split}')
        for image, label, sha, classes in rows[split]:
            if sha in seen:
                raise ValueError(f'Identical decoded images in dataset: {seen[sha]} and {split}/{image.name}')
            seen[sha] = f'{split}/{image.name}'
            if split == 'val':
                val_classes.update(classes)
    if val_classes != set(range(len(CLASSES))):
        raise ValueError('Validation set must include all six classes')
    evaluation = [(split, sha, hashlib.sha256(label.read_bytes()).hexdigest())
                  for split in ('val','test') for _, label, sha, _ in rows[split]]
    fingerprint = hashlib.sha256(json.dumps(sorted(evaluation)).encode()).hexdigest()
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError('Fixed validation/test set has changed; automatic learning blocked')
    for split in ('train','val'):
        (dest / split / 'images').mkdir(parents=True)
        (dest / split / 'labels').mkdir(parents=True)
        for index, (image, label, _, _) in enumerate(rows[split]):
            name = f'base_{index:06d}'
            shutil.copy2(image, dest / split / 'images' / (name + image.suffix.lower()))
            shutil.copy2(label, dest / split / 'labels' / (name + '.txt'))
    initialize()
    approved = []
    skipped = []
    with closing(sqlite3.connect(DB)) as conn:
        cursor = conn.execute("SELECT sha,image_png,width,height,labels_json FROM samples WHERE status='approved' ORDER BY sha")
        for sha, png, w, h, raw in cursor:
            if sha in seen:
                skipped.append({'sha':sha, 'reason':seen[sha]})
                continue
            frame = cv2.imdecode(np.frombuffer(png,dtype=np.uint8),cv2.IMREAD_COLOR)
            if frame is None or frame.shape[:2] != (h,w) or image_fingerprint(frame) != sha:
                raise ValueError(f'Corrupt reviewed image: {sha}')
            lines = []
            for row in json.loads(raw):
                c = CLASSES.index(normalize_class(row['class']))
                x1,y1,x2,y2 = map(float,row['bbox_xyxy'])
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise ValueError(f'Invalid approved annotation: {sha}')
                lines.append(f'{c} {(x1+x2)/(2*w):.9f} {(y1+y2)/(2*h):.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}')
            stem = 'feedback_' + sha
            (dest/'train'/'images'/f'{stem}.png').write_bytes(png)
            (dest/'train'/'labels'/f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
            approved.append(sha)
    new = sorted(set(approved)-set(consumed))
    if len(new) < threshold:
        raise ValueError(f'Only {len(new)} new unique non-overlapping samples; need {threshold}')
    config = {'path': str(dest.resolve()), 'train':'train/images','val':'val/images','names':dict(enumerate(names))}
    yaml_path = dest/'data.yaml'
    yaml_path.write_text(yaml.safe_dump(config,sort_keys=False))
    manifest = {'base_train':len(rows['train']), 'val':len(rows['val']), 'reserved_test':len(rows['test']),
                'approved':approved,'new':new,'excluded_duplicates':skipped,'evaluation_fingerprint':fingerprint}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return yaml_path, manifest
STEEL_PYTHON_1

cat > "$TMP/training_worker.py" <<'STEEL_PYTHON_2'
"""One offline champion/challenger job per systemd timer invocation."""
import gc
import importlib.metadata
import json
import math
import os
import shutil
import statistics
import subprocess
import time
import traceback
import uuid
from pathlib import Path

# Set before importing Ultralytics or Torch.
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from . import model_registry as reg
from .dataset_builder import approved_count, build, pictures, parse_names


class Deferred(RuntimeError):
    pass


def report(status, detail, **extra):
    value = {'status':status, 'detail':str(detail), 'updated_at':reg.now(), **extra}
    reg.write_json(reg.STATUS,value)
    print(json.dumps(value), flush=True)
    return value


def resources(start=False):
    import psutil
    if not reg.idle_for(90):
        raise Deferred('Inspection activity detected; training deferred or interrupted')
    temp = int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000
    if temp >= 75:
        raise Deferred(f'CPU temperature {temp:.1f} C exceeds limit')
    result = subprocess.check_output(['vcgencmd','get_throttled'],text=True,timeout=5).strip()
    flags = int(result.split('=')[-1],16)
    if flags & 0xf:
        raise Deferred(f'Current power/thermal throttling: {result}')
    available = psutil.virtual_memory().available / 1024**3
    if available < (3.0 if start else 1.0):
        raise Deferred(f'Insufficient available RAM: {available:.2f} GiB')
    if shutil.disk_usage(reg.WORK).free < 5*1024**3:
        raise Deferred('Less than 5 GiB storage remains')


def install_callbacks(model):
    last = [0.0]
    def guard(_):
        current = time.monotonic()
        if current - last[0] >= 8:
            resources()
            last[0] = current
    for event in ('on_train_batch_end','on_val_batch_end','on_train_epoch_end'):
        model.add_callback(event,guard)


def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.0


def metrics_for(path, data, job, tag):
    import cv2
    import yaml
    from ultralytics import YOLO
    model=YOLO(str(path),task='detect')
    install_callbacks(model)
    result=model.val(data=str(data),split='val',imgsz=320,device='cpu',batch=1,workers=0,
                     conf=0.001,iou=0.7,rect=False,plots=False,save_json=False,
                     project=str(job/'validation'),name=tag,exist_ok=True,verbose=False)
    box=result.box
    metrics={'map50':float(box.map50),'map5095':float(box.map)}
    # Check the deployed model at the dashboard's normal 0.30 operating confidence.
    names=parse_names(yaml.safe_load(data.read_text())['names'])
    images=pictures(data.parent/'val'/'images')
    tp=[0]*6; fp=[0]*6; fn=[0]*6; times=[]
    for p in images[:3]:
        resources()
        model.predict(cv2.imread(str(p)),imgsz=320,conf=0.30,iou=0.7,device='cpu',verbose=False)
    for p in images:
        resources()
        frame=cv2.imread(str(p)); h,w=frame.shape[:2]
        start=time.perf_counter()
        output=model.predict(frame,imgsz=320,conf=0.30,iou=0.7,device='cpu',verbose=False)[0]
        times.append((time.perf_counter()-start)*1000)
        if parse_names(output.names) != names:
            # Hyphen/underscore variants are allowed, but ordering is not.
            from .feedback_store import normalize_class
            if list(map(normalize_class,parse_names(output.names))) != list(map(normalize_class,names)):
                raise ValueError('NCNN model class order mismatch')
        truth=[]
        for line in (data.parent/'val'/'labels'/p.with_suffix('.txt').name).read_text().splitlines():
            if not line.strip(): continue
            c,x,y,bw,bh=map(float,line.split()); truth.append((int(c),((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)))
        boxes=output.boxes
        predictions=sorted(zip(boxes.conf.cpu().tolist(),boxes.cls.cpu().tolist(),boxes.xyxy.cpu().tolist()),reverse=True)
        matched=set()
        for score, cls, coords in predictions:
            c=int(cls)
            if not 0<=c<6: raise ValueError('Unexpected model class')
            options=[(iou(coords,t),i) for i,(tc,t) in enumerate(truth) if tc==c and i not in matched]
            best=max(options,default=(0,-1))
            if best[0]>=0.5: tp[c]+=1; matched.add(best[1])
            else: fp[c]+=1
        for i,(c,_) in enumerate(truth):
            if i not in matched: fn[c]+=1
    total_tp,total_fp,total_fn=map(sum,(tp,fp,fn))
    metrics.update(precision=total_tp/max(1,total_tp+total_fp),recall=total_tp/max(1,total_tp+total_fn),
                   f1=2*total_tp/max(1,2*total_tp+total_fp+total_fn),
                   recall_by_class={n:tp[i]/max(1,tp[i]+fn[i]) for i,n in enumerate(names)},
                   median_ms=statistics.median(times), images=len(images),tp=total_tp,fp=total_fp,fn=total_fn)
    if not all(math.isfinite(metrics[k]) for k in ('map50','map5095','precision','recall','f1','median_ms')):
        raise ValueError('Non-finite validation result')
    del model,result,output
    gc.collect()
    return metrics


def compare(old,new,cfg):
    checks={'map50_improved':new['map50']-old['map50']>=cfg['min_map50_gain'],
            'map5095_not_worse':new['map5095']>=old['map5095'],
            'precision_not_worse':new['precision']>=old['precision']-cfg['max_pr_drop'],
            'recall_not_worse':new['recall']>=old['recall']-cfg['max_pr_drop'],
            'f1_not_worse':new['f1']>=old['f1'],
            'class_recall_not_worse':all(new['recall_by_class'][k]>=v-cfg['max_class_recall_drop'] for k,v in old['recall_by_class'].items()),
            'latency_within_budget':new['median_ms']<=old['median_ms']*cfg['max_latency_ratio']}
    return checks


def run():
    with reg.exclusive():
        cfg=reg.read_json(reg.CONFIG)
        state=reg.read_state()
        for record in (reg.WORK/'training_jobs').glob('*/job.json'):
            previous=reg.read_json(record)
            if previous.get('status')=='running':
                committed=state['versions'].get(previous['id'])
                previous.update(status=committed['decision'] if committed else 'Interrupted',finished_at=reg.now())
                reg.write_json(record,previous)
        active=state['versions'][state['active']]
        if not cfg['enabled']:
            report('Disabled','Automatic training disabled in backend configuration'); return
        root=Path(cfg['base_dataset'])
        seed=reg.REG/active['id']/'best.pt' if active.get('pt_sha') else Path(cfg['seed_checkpoint'])
        blockers=[]
        if not seed.is_file() or seed.suffix.lower()!='.pt': blockers.append(f'Trusted trainable checkpoint missing: {seed}')
        if not (root/'data.yaml').is_file(): blockers.append('Original data.yaml missing')
        for split in ('train','val','test'):
            if not pictures(root/split/'images') or not list((root/split/'labels').rglob('*.txt')):
                blockers.append(f'Original {split} images/labels missing: {root/split}')
        try:
            installed=importlib.metadata.version('pnnx')
            if installed != cfg['pnnx_version']: blockers.append(f'PNNX {cfg["pnnx_version"]} required; installed {installed}')
        except importlib.metadata.PackageNotFoundError:
            blockers.append(f'PNNX {cfg["pnnx_version"]} not installed')
        count=approved_count()
        if blockers:
            report('Blocked','; '.join(blockers),approved=count,threshold=cfg['threshold']); return
        if count-len(state['consumed'])<cfg['threshold']:
            report('Waiting for data','Not enough new approved images',approved=count,threshold=cfg['threshold']); return
        resources(start=True)
        reg.verify_entry(active)
        # Never load untrusted checkpoints: seed.pt must be manually supplied by device administrator.
        job_id='v'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
        job=reg.WORK/'training_jobs'/job_id; job.mkdir(parents=True)
        snapshot=reg.WORK/'dataset_versions'/job_id
        info={'id':job_id,'parent':active['id'],'started_at':reg.now(),'status':'running'}
        reg.write_json(job/'job.json',info)
        try:
            report('Preparing dataset',job_id,job=job_id,approved=count,threshold=cfg['threshold'])
            data,manifest=build(root,snapshot,state['consumed'],cfg['threshold'],state.get('evaluation_fingerprint'))
            info['manifest']=str(snapshot/'manifest.json')
            if state.get('evaluation_fingerprint') is None:
                state['evaluation_fingerprint']=manifest['evaluation_fingerprint']
                reg.write_json(reg.STATE,state)
            resources(start=True)
            report('Training','CPU fine-tuning',job=job_id,approved=count)
            import torch
            from ultralytics import YOLO, settings
            torch.set_num_threads(2)
            try: torch.set_num_interop_threads(1)
            except RuntimeError: pass
            for key in ('sync','tensorboard','wandb','mlflow','clearml','comet','hub','neptune','raytune'):
                if key in settings: settings.update({key:False})
            model=YOLO(str(seed),task='detect')
            install_callbacks(model)
            model.train(data=str(data),device='cpu',imgsz=320,epochs=cfg['epochs'],batch=2,workers=0,
                        freeze=cfg['freeze'],amp=False,cache=False,patience=3,optimizer='SGD',lr0=0.001,
                        mosaic=0.0,mixup=0.0,plots=False,seed=42,project=str(job),name='fit',exist_ok=True)
            checkpoint=Path(model.trainer.best)
            if not checkpoint.is_file(): raise RuntimeError('Training did not produce best.pt')
            del model
            gc.collect()
            resources()
            report('Exporting','Exporting candidate to NCNN',job=job_id)
            candidate_model=YOLO(str(checkpoint),task='detect')
            exported=Path(candidate_model.export(format='ncnn',imgsz=320,device='cpu',batch=1))
            del candidate_model
            gc.collect()
            reg.artifact_hashes(exported)
            resources()
            report('Evaluating','Compare active and candidate NCNN on fixed validation set',job=job_id)
            old=metrics_for(reg.entry_path(active),data,job,'champion')
            new=metrics_for(exported,data,job,'challenger')
            checks=compare(old,new,cfg)
            passed=all(checks.values())
            reg.write_json(job/'comparison.json',{'champion':old,'challenger':new,'checks':checks,'passed':passed})
            resources()
            target=reg.REG/job_id
            target.mkdir()
            shutil.copytree(exported,target/'detector_ncnn_model')
            shutil.copy2(checkpoint,target/'best.pt')
            entry={'id':job_id,'created_at':reg.now(),'parent':active['id'],'ever_active':passed,
                   'decision':'promoted' if passed else 'rejected','hashes':reg.artifact_hashes(target/'detector_ncnn_model'),
                   'pt_sha':reg.digest(target/'best.pt'),'metrics':new,'checks':checks,'job':str(job),
                   'new_samples':len(manifest['new'])}
            if entry['hashes'] != reg.artifact_hashes(exported) or entry['pt_sha'] != reg.digest(checkpoint):
                raise RuntimeError('Registered candidate copy differs from validated artifacts')
            reg.verify_entry(entry)
            # Flush copied artifacts before committing the registry pointer.
            for file in target.rglob('*'):
                if file.is_file():
                    with file.open('rb') as stream: os.fsync(stream.fileno())
            for directory in (target/'detector_ncnn_model',target,reg.REG):
                fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
            state['versions'][active['id']]['metrics']=old
            state['versions'][job_id]=entry
            state['consumed']=sorted(set(state['consumed'])|set(manifest['new']))
            if passed:
                resources()
                state['active']=job_id
            state['events'].append({'time':reg.now(),'action':entry['decision'],'from':active['id'],'to':job_id,'checks':checks})
            # Single atomic commit: no partial promotion or overwritten active folder.
            reg.write_json(reg.STATE,state)
            info.update(status=entry['decision'],finished_at=reg.now(),checks=checks)
            reg.write_json(job/'job.json',info)
            report(entry['decision'].capitalize(),'Candidate '+('activated' if passed else 'rejected; previous model retained'),job=job_id,checks=checks)
        except Exception as exc:
            status='Deferred' if isinstance(exc,Deferred) else 'Failed'
            info.update(status=status,finished_at=reg.now(),error=str(exc))
            reg.write_json(job/'job.json',info)
            report(status,str(exc),job=job_id)
            traceback.print_exc()
            # Failed/deferred samples remain eligible. Preserve evidence but remove bulk incomplete dataset.
            if snapshot.exists() and not (job/'comparison.json').exists(): shutil.rmtree(snapshot)


if __name__=='__main__':
    try:
        run()
    except Deferred as exc:
        report('Deferred',str(exc))
    except Exception as exc:
        report('Failed',str(exc))
        traceback.print_exc()
        raise
STEEL_PYTHON_2

cat > "$TMP/model_history.py" <<'STEEL_PYTHON_3'
"""Operator-facing model history and manual rollback."""
import sys
from pathlib import Path
import streamlit as st
SRC=Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from learning import model_registry as reg
from learning.feedback_store import statistics


@st.fragment(run_every='15s')
def render_model_history(loaded_path):
    state=reg.read_state()
    active=state['versions'][state['active']]
    cfg=reg.read_json(reg.CONFIG)
    status=reg.read_json(reg.STATUS) if reg.STATUS.exists() else {'status':'Waiting','detail':'Training timer has not run yet'}
    count=statistics()['approved']
    st.subheader('Model history')
    a,b,c=st.columns(3)
    a.metric('Active version',active['id'],border=True)
    b.metric('Approved samples',count,border=True)
    c.metric('Training status',status['status'],border=True)
    st.caption(str(status.get('detail','')))
    st.caption(f"Automatic CPU fine-tuning: {cfg['threshold']} new approved images; {cfg['epochs']} epochs. All computation runs locally.")
    if str(reg.active_path()) != str(loaded_path):
        st.info('A model version changed. Reloading when inspection is idle.')
        if not st.session_state.get('pi_live_running',False) and reg.idle_for(15):
            st.rerun(scope='app')
    rows=[]
    for version,entry in reversed(list(state['versions'].items())):
        m=entry.get('metrics') or {}
        rows.append({'Version':version,'Status':'Active' if version==state['active'] else ('Archived' if entry['ever_active'] else 'Rejected'),
                     'mAP50 (%)':round(100*m['map50'],2) if 'map50' in m else None,
                     'Recall (%)':round(100*m['recall'],2) if 'recall' in m else None,
                     'Median latency (ms)':round(m['median_ms'],1) if 'median_ms' in m else None,'Created UTC':entry['created_at']})
    st.dataframe(rows,hide_index=True,width='stretch')
    options=[k for k,v in state['versions'].items() if v['ever_active'] and k!=state['active']]
    if options:
        with st.form('model_rollback_form'):
            version=st.selectbox('Restore previously active model',options)
            confirmed=st.checkbox('I confirm the model rollback')
            submitted=st.form_submit_button('Rollback model')
        if submitted:
            if not confirmed: st.warning('Confirm the rollback first.')
            elif st.session_state.get('pi_live_running',False): st.warning('Stop live inspection first.')
            else:
                try: reg.rollback(version)
                except Exception as exc: st.error(str(exc))
                else: st.rerun(scope='app')
    else: st.caption('Rollback becomes available after the first successful model replacement.')
    with st.expander('Model decisions and evaluation history'):
        st.json(list(reversed(state['events'])))
    jobs=[]
    for path in sorted((reg.WORK/'training_jobs').glob('*/job.json'),reverse=True)[:30]:
        try:
            record=reg.read_json(path)
            jobs.append({k:record.get(k) for k in ('id','parent','status','started_at','finished_at','error')})
        except (OSError,ValueError): continue
    with st.expander('Recent training jobs'):
        if jobs: st.dataframe(jobs,hide_index=True,width='stretch')
        else: st.caption('No training jobs yet.')
STEEL_PYTHON_3

cat > "$TMP/install_patch.py" <<'STEEL_PYTHON_4'
"""Patch the current Steel Inspector dashboard, preserving old source."""
import ast
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

root=Path.home()/'Industrial_Defect_Detection'
src=root/'src'
work=Path.home()/'soc-retraining'
package=Path(sys.argv[1])
app=src/'dashboard/app_pi.py'
helper=src/'dashboard/dashboard_ui.py'
qc=src/'inference/qc_llama.py'
browser=src/'dashboard/browser_live.py'
files={src/'learning'/n:(package/n).read_text() for n in ('model_registry.py','dataset_builder.py','training_worker.py')}
files[src/'dashboard/model_history.py']=(package/'model_history.py').read_text()

def top_function(s,name):
    matches=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(matches)!=1: raise RuntimeError(f'Expected one function {name}')
    return matches[0]

def decorate(s,name):
    if f'@inspection_activity\ndef {name}(' in s: return s
    n=top_function(s,name)
    if n.decorator_list: raise RuntimeError(f'Unexpected decorators on {name}')
    lines=s.splitlines(keepends=True)
    lines.insert(n.lineno-1, 'import sys as _learning_sys\nfrom pathlib import Path as _LearningPath\n_learning_src = str(_LearningPath(__file__).resolve().parents[1])\nif _learning_src not in _learning_sys.path:\n    _learning_sys.path.insert(0, _learning_src)\nfrom learning.model_registry import inspection_activity\n\n@inspection_activity\n')
    return ''.join(lines)

s=app.read_text()
if 'from model_history import render_model_history' not in s:
    assert 'render_learning_feedback' in s and 'steel_inspection' in s, 'Part 1 and final dark dashboard required'
    anchor='from learning_feedback import render_learning_feedback'
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+'\nfrom model_history import render_model_history\nfrom learning.model_registry import active_path as active_model_path',1)
    tree=ast.parse(s)
    model_assign=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model_path' for t in n.targets)]
    tabs=[n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='tabs']
    if len(model_assign)!=1 or len(tabs)!=1: raise RuntimeError('Unexpected model or tab layout; app left unchanged')
    m,t=model_assign[0],tabs[0]
    assert 'st.sidebar.text_input' in ast.get_source_segment(s,m)
    assert len(t.targets)==1 and isinstance(t.targets[0],ast.Tuple) and len(t.targets[0].elts)==3
    assert isinstance(t.value.args[0],(ast.List,ast.Tuple)) and len(t.value.args[0].elts)==3
    lines=s.splitlines(keepends=True)
    targets=[e.id for e in t.targets[0].elts]
    labels=[ast.literal_eval(e) for e in t.value.args[0].elts]
    updates=[(m.lineno,m.end_lineno,"model_path = str(active_model_path())\nst.sidebar.caption('Managed model: ' + Path(model_path).parent.name)\n"),
             (t.lineno,t.end_lineno,', '.join(targets+['models_tab'])+' = st.tabs('+repr(labels+['Model history'])+')\n')]
    for start,end,replacement in sorted(updates,reverse=True): lines[start-1:end]=[replacement]
    s=''.join(lines)
    s=s.replace("@st.cache_resource(show_spinner='Loading NCNN model on Raspberry Pi CPU…')", "@st.cache_resource(show_spinner='Loading NCNN model on Raspberry Pi CPU…', max_entries=2)")
    s+='\n\nwith models_tab:\n    render_model_history(str(Path(model_path).resolve()))\n'
files[app]=s
files[helper]=decorate(helper.read_text(),'predict_for_dashboard')
files[qc]=decorate(qc.read_text(),'generate_grounded_qc_answer')
s=browser.read_text()
if 'key="pi-browser-live-yolo"' in s:
    s=s.replace('key="pi-browser-live-yolo"','key="pi-browser-live-yolo-" + _model_key(str(model_path).encode()).hexdigest()[:16]',1)
    n=top_function(s,'render_browser_live')
    lines=s.splitlines(keepends=True)
    lines.insert(n.lineno-1,'from hashlib import sha256 as _model_key\n\n')
    s=''.join(lines)
elif '_model_key(str(model_path)' not in s:
    raise RuntimeError('Unexpected browser camera key; no application files changed')
files[browser]=s
for p,code in files.items(): compile(code,str(p),'exec')
backup=Path.home()/'soc-project-backups'/('auto_learning_full_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
backup.mkdir(parents=True)
original={p:p.read_bytes() if p.exists() else None for p in files}
for p,content in original.items():
    if content is not None:
        target=backup/p.relative_to(root); target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(content)
try:
    for p,code in files.items():
        p.parent.mkdir(parents=True,exist_ok=True); p.write_text(code); print('Installed',p.relative_to(root))
    sys.path.insert(0,str(src))
    from learning import model_registry as reg
    reg.initialize(root/'models/exported/best_ncnn_model')
    if not reg.CONFIG.exists():
        cfg={'enabled':True,'base_dataset':str(root/'datasets/processed/NEU-DET'),'seed_checkpoint':str(work/'config/seed.pt'),
             'threshold':50,'epochs':5,'freeze':10,'min_map50_gain':0.01,'max_pr_drop':0.0,'max_class_recall_drop':0.0,
             'max_latency_ratio':1.10,'pnnx_version':'20260526'}
        reg.write_json(reg.CONFIG,cfg)
    print('Active model:',reg.active_path())
except BaseException:
    for p,content in original.items():
        if content is None: p.unlink(missing_ok=True)
        else: p.write_bytes(content)
    raise
print('Source backup:',backup)
STEEL_PYTHON_4


# Compile the embedded Python before modifying the working application.
"$CPU_PYTHON" -m py_compile "$TMP"/*.py
sudo systemctl stop steel-inspector.service
APP_STOPPED=1
"$CPU_PYTHON" "$TMP/install_patch.py" "$TMP"

mkdir -p "$WORK/training_jobs" "$WORK/dataset_versions" "$WORK/training_logs" "$WORK/config"
UNIT_BACKUP="$HOME/soc-project-backups/learning_units_$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p "$UNIT_BACKUP"
for name in steel-learning.service steel-learning.timer; do
    if sudo test -f "/etc/systemd/system/$name"; then sudo cp "/etc/systemd/system/$name" "$UNIT_BACKUP/"; fi
done
cat > "$TMP/steel-learning.service" <<EOF
[Unit]
Description=Steel Inspector offline CPU model learning
After=local-fs.target steel-inspector.service

[Service]
Type=oneshot
User=$(id -un)
Group=$(id -gn)
WorkingDirectory=$PROJECT
Environment=HOME=$HOME
Environment=PATH=$CPU_ENV/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=$PROJECT/src
Environment=PYTHONUNBUFFERED=1
Environment=YOLO_AUTOINSTALL=false
Environment=YOLO_OFFLINE=true
Environment=OMP_NUM_THREADS=2
Environment=MKL_NUM_THREADS=2
ExecStart=$CPU_PYTHON -m learning.training_worker
Nice=15
CPUQuota=200%
MemoryHigh=3500M
MemoryMax=4500M
PrivateNetwork=true
NoNewPrivileges=true
TimeoutStartSec=infinity
TimeoutStopSec=30
EOF
cat > "$TMP/steel-learning.timer" <<'EOF'
[Unit]
Description=Check for new verified steel inspection samples

[Timer]
OnBootSec=3min
OnUnitInactiveSec=10min
Unit=steel-learning.service
AccuracySec=30s

[Install]
WantedBy=timers.target
EOF
sudo systemd-analyze verify "$TMP/steel-learning.service" "$TMP/steel-learning.timer"
sudo install -m 644 "$TMP/steel-learning.service" /etc/systemd/system/steel-learning.service
sudo install -m 644 "$TMP/steel-learning.timer" /etc/systemd/system/steel-learning.timer
sudo systemctl daemon-reload
sudo systemctl start steel-inspector.service
APP_STOPPED=0
sudo systemctl enable --now steel-learning.timer
sudo systemctl start --no-block steel-learning.service

# Basic installation verification, not a real training run.
echo '=== DASHBOARD SERVICE ==='
systemctl is-active steel-inspector.service
systemctl is-enabled steel-inspector.service
echo '=== LEARNING TIMER ==='
systemctl is-enabled steel-learning.timer
systemctl list-timers steel-learning.timer --no-pager
sleep 3
echo '=== LEARNING STATUS ==='
if [ -f "$WORK/training_status.json" ]; then cat "$WORK/training_status.json"; echo; fi
echo '=== BACKEND CONFIG ==='
cat "$WORK/config/learning.json"; echo
echo '=== SOURCE STATUS ==='
cd "$PROJECT"
git status --short
echo 'INSTALLATION COMPLETE: training depends on valid local assets and approved data.'
echo 'Original model directory preserved. Review Model history in the dashboard.'
