#!/usr/bin/env bash
# Final feature update, run on Pi as pi. Does not force model training or reboot.
set -euo pipefail
[ "$(id -un)" = pi ] && [ "$(uname -m)" = aarch64 ] || { echo 'Run as pi on Raspberry Pi ARM64'; exit 1; }
ROOT="$HOME/Industrial_Defect_Detection"
PY="$HOME/soc-venvs/cpu-validation-BdkS2KBp/.venv/bin/python"
[ -x "$PY" ]
[ -f "$ROOT/src/dashboard/operator_ui.py" ]
[ -f "$ROOT/src/learning/model_registry.py" ]
systemctl cat steel-inspector.service >/dev/null
systemctl cat steel-learning.timer >/dev/null
sudo -v
TMP=$(mktemp -d)
BACKUP="$HOME/soc-project-backups/final_features_$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p "$BACKUP"
COMMITTED=0; STOPPED=0; TIMER_STOPPED=0
cleanup() {
 code=$?
 if [ "$COMMITTED" = 0 ] && [ -f "$BACKUP/manifest.json" ]; then
   sudo systemctl stop steel-inspector.service || true
   "$PY" - "$ROOT" "$BACKUP" <<'RESTORE' || true
import json,sys
from pathlib import Path
root,backup=map(Path,sys.argv[1:])
for name,existed in json.loads((backup/'manifest.json').read_text()).items():
 p=root/name
 if existed: p.write_bytes((backup/name).read_bytes())
 else: p.unlink(missing_ok=True)
print('Source backups restored.')
RESTORE
   STOPPED=1
 fi
 if [ "$STOPPED" = 1 ]; then sudo systemctl start steel-inspector.service || true; fi
 if [ "$TIMER_STOPPED" = 1 ]; then sudo systemctl start steel-learning.timer || true; fi
 rm -rf "$TMP"
 exit "$code"
}
trap cleanup EXIT

cat > "$TMP/final_features.py" <<'STEEL_FINAL_0'
"""Local operator tools; optional research only uses public HTTPS sources."""
import csv, io, json, math, os, re, shutil, socket, ssl, http.client, ipaddress, subprocess, sqlite3, time
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from xml.sax.saxutils import escape
import cv2
import numpy as np
import psutil
import streamlit as st
from learning import model_registry as reg
from learning import operator_backend as ops
from learning.feedback_store import DB as LEARN_DB, collect_image, statistics
from learning_feedback import render_learning_feedback
from learning.dataset_builder import pictures

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'datasets/processed/NEU-DET'
# Fixed English class names retained as data; translated presentation only.

@st.fragment(run_every='10s')
def render_health():
    st.subheader('System health')
    mem=psutil.virtual_memory(); disk=shutil.disk_usage(reg.WORK)
    try: temp=int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000
    except (OSError, ValueError): temp=None
    a,b,c,d=st.columns(4)
    a.metric('CPU utilization',f'{psutil.cpu_percent(interval=0.1):.1f}%',border=True)
    b.metric('Memory utilization',f'{mem.percent:.1f}%',border=True)
    c.metric('CPU temperature',f'{temp:.1f} °C' if temp is not None else 'Unavailable',border=True)
    d.metric('Free storage',f'{disk.free/1024**3:.1f} GiB',border=True)
    try:
        raw=subprocess.check_output(['vcgencmd','get_throttled'],text=True,timeout=3).strip(); flags=int(raw.split('=')[-1],16)
        names=['Undervoltage','Frequency capped','Throttled','Soft temperature limit']
        st.write(raw)
        for i,n in enumerate(names):
            if flags & (1<<i): st.warning(n+' — current')
            if flags & (1<<(i+16)): st.caption(n+' — occurred since boot')
        if flags==0: st.success('No power or throttling flags reported.')
    except Exception as exc: st.caption('Power flags unavailable: '+str(exc))
    if temp is not None and temp>=75: st.warning('High temperature; defer training.')
    st.caption('Utilization and throttling are not measurements of electrical power in watts.')
    for unit in ('steel-inspector.service','steel-learning.service','steel-learning.timer'):
        try: status=subprocess.run(['systemctl','show','-p','ActiveState','--value',unit],capture_output=True,text=True,timeout=3).stdout.strip()
        except Exception: status='unknown'
        st.write(f'{unit}: {status}')


def saved_rows(db,limit=2000):
    with closing(sqlite3.connect(Path(db).resolve().as_uri()+'?mode=ro',uri=True,timeout=5)) as c:
        c.row_factory=sqlite3.Row
        return [dict(r) for r in c.execute('SELECT * FROM pi_inspections ORDER BY timestamp_utc DESC LIMIT ?', (limit,))]


def render_analytics(db):
    st.subheader('Inspection analytics')
    rows=saved_rows(db)
    classes=Counter(); reviews=Counter(); days=Counter()
    for r in rows:
        reviews[r['priority']]+=1; days[r['timestamp_utc'][:10]]+=1
        for d in json.loads(r['detections_json']): classes[str(d.get('class','unknown'))]+=1
    a,b,c=st.columns(3)
    a.metric('Saved inspections',len(rows),border=True); b.metric('Detected regions',sum(classes.values()),border=True); c.metric('High review',reviews['HIGH REVIEW'],border=True)
    st.caption('Latest 2,000 saved inspections only; counts are not production defect rates or yield.')
    if not rows: st.info('No saved inspections yet.'); return
    import pandas as pd
    st.write('Defect distribution');
    if classes: st.bar_chart(pd.Series(dict(classes),dtype=int))
    else: st.caption('No detected regions in saved inspections.')
    st.write('Inspection activity by day'); st.line_chart(pd.Series(dict(sorted(days.items())),dtype=int))
    st.write('Review categories'); st.dataframe([{'Review':k,'Count':v} for k,v in reviews.items()],hide_index=True)


def pdf_bytes(row,out):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    buf=io.BytesIO(); styles=getSampleStyleSheet(); story=[]
    def para(v,style='BodyText'):
        story.append(Paragraph(escape(str(v)).replace('\n','<br/>'),styles[style])); story.append(Spacer(1,7))
    para('Steel Inspector | Inspection report','Title')
    para('Research prototype · Human QC verification required','Heading2')
    for label,key in [('Inspection ID','id'),('Captured UTC','timestamp_utc'),('Image','filename'),('Model','model'),('Review','priority'),('Confidence threshold','conf'),('Input size','imgsz')]: para(f'{label}: {row.get(key,"—")}')
    base=Path(out).resolve(); p=base/(Path(str(row.get('annotated_file',''))).name)
    if p.is_file() and p.resolve().is_relative_to(base):
        w,h=ImageReader(str(p)).getSize(); scale=min(450/w,240/h); story.append(Image(str(p),width=w*scale,height=h*scale)); story.append(Spacer(1,10))
    para('Detected regions','Heading2')
    table=[['#','Class','Confidence','Box (pixels)']]
    for i,d in enumerate(json.loads(row['detections_json']),1): table.append([str(i),str(d.get('class')),f"{float(d.get('confidence',0))*100:.1f}%",str(d.get('bbox_xyxy'))])
    t=Table(table,colWidths=[28,115,70,260],repeatRows=1); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#18304e')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTSIZE',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf2f7')]) ])); story.append(t)
    para('Stored inspection guidance','Heading2'); para(row.get('recommendation',''))
    para('Confidence is not severity. No detections does not establish acceptance. Possible causes require independent confirmation.')
    SimpleDocTemplate(buf,pagesize=A4,leftMargin=48,rightMargin=48,topMargin=48,bottomMargin=48,title='Steel Inspector inspection report').build(story)
    return buf.getvalue()


def render_reports(db,out):
    st.subheader('Inspection reports')
    rows=saved_rows(db,300)
    if not rows: st.info('Save an inspection first.'); return
    by_id={r['id']:r for r in rows}
    key=st.selectbox('Select saved inspection',list(by_id),format_func=lambda k:f"{by_id[k]['filename']} · {by_id[k]['timestamp_utc']} · {k[:8]}",key='final_pdf_select')
    if st.button('Prepare PDF report',key='final_pdf_prepare'):
        try: st.session_state['final_pdf']=(key,pdf_bytes(by_id[key],out))
        except Exception as e: st.error(str(e))
    prepared=st.session_state.get('final_pdf')
    if prepared and prepared[0]==key: st.download_button('Download PDF report',prepared[1],f'inspection_{key}.pdf','application/pdf',key='final_pdf_download')


def descriptor(frame):
    g=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(96,96),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    gx=cv2.Sobel(g,cv2.CV_32F,1,0); gy=cv2.Sobel(g,cv2.CV_32F,0,1)
    def hist(a,hi):
        h=np.histogram(np.clip(a,0,hi),bins=16,range=(0,hi))[0].astype(np.float32); return h/max(h.sum(),1)
    return np.concatenate([hist(g,1),hist(np.sqrt(gx*gx+gy*gy),4),[g.mean(),g.std()] ]).astype(np.float32)


def build_reference():
    train=[]; val=[]
    for split,target in [('train',train),('val',val)]:
        for p in pictures(BASE/split/'images'):
            im=cv2.imread(str(p));
            if im is None: raise ValueError(f'Unreadable {p}')
            target.append(descriptor(im))
    if len(train)<100 or len(val)<50: raise ValueError('Reference data missing')
    x=np.stack(train); y=np.stack(val); scale=np.maximum(x.std(axis=0),.025); x=x/scale; y=y/scale
    scores=[float(np.sqrt(np.mean((x-v)**2,axis=1)).min()) for v in y]
    dest=reg.WORK/'texture_reference.npz'; temp=dest.with_suffix('.tmp')
    with temp.open('wb') as f: np.savez_compressed(f,reference=x,scale=scale,threshold=np.quantile(scores,.95),train_count=len(x),val_count=len(y))
    os.replace(temp,dest)
    return len(x),len(y),float(np.quantile(scores,.95))


@st.cache_resource(max_entries=1)
def load_reference(path,mtime):
    with np.load(path,allow_pickle=False) as data: return {k:data[k].copy() for k in data.files}


def novelty(frame):
    p=reg.WORK/'texture_reference.npz'
    if not p.exists(): return None
    r=load_reference(str(p),p.stat().st_mtime_ns)
    v=descriptor(frame)/r['scale']; score=float(np.sqrt(np.mean((r['reference']-v)**2,axis=1)).min())
    return score,float(r['threshold'])


def render_novelty(frame):
    try: result=novelty(frame)
    except Exception as exc: st.caption('Texture check unavailable: '+str(exc)); return
    if result is None: st.caption('Texture reference unavailable.'); return
    score,threshold=result
    if score>threshold: st.warning(f'Unfamiliar texture: review required (distance {score:.3f}; reference threshold {threshold:.3f}).')
    else: st.caption(f'Texture distance {score:.3f} · reference threshold {threshold:.3f}')
    st.caption('Experimental out-of-distribution screening, not a validated unknown-defect detector or normal-surface classifier.')


class PageText(HTMLParser):
    def __init__(self): super().__init__(convert_charrefs=True); self.skip=0; self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style','noscript','svg','nav','header','footer'): self.skip+=1
        if tag in ('p','li','h1','h2','h3','br'): self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in ('script','style','noscript','svg','nav','header','footer'): self.skip=max(0,self.skip-1)
        if tag in ('p','li','h1','h2','h3'): self.parts.append('\n')
    def handle_data(self,data):
        if not self.skip: self.parts.append(data)


# No redirects, proxies, credentials, private IPs, arbitrary ports or unchecked DNS rebinding.
class PublicHTTPS(http.client.HTTPSConnection):
    def connect(self):
        infos=socket.getaddrinfo(self.host,443,type=socket.SOCK_STREAM)
        addresses=[x[4][0] for x in infos]
        if not addresses or any(not ipaddress.ip_address(a).is_global for a in addresses): raise ValueError('Non-public address refused')
        raw=socket.create_connection((addresses[0],443),timeout=5)
        try: self.sock=ssl.create_default_context().wrap_socket(raw,server_hostname=self.host)
        except BaseException: raw.close(); raise


def page_extract(url,words):
    u=urlsplit(url)
    if u.scheme!='https' or not u.hostname or u.username or u.password or u.port not in (None,443): raise ValueError('Only public HTTPS pages supported')
    c=PublicHTTPS(u.hostname,443,timeout=5)
    try:
        c.request('GET',(u.path or '/')+('?' +u.query if u.query else ''),headers={'User-Agent':'SteelInspector/1.0 (operator-requested research)','Accept':'text/html','Accept-Encoding':'identity'})
        response=c.getresponse()
        if response.status!=200 or 'text/html' not in response.getheader('Content-Type','').lower(): raise ValueError(f'Unsupported HTTP response {response.status}')
        raw=response.read(400_001)
        if len(raw)>400_000: raise ValueError('Page exceeds retrieval limit')
    finally: c.close()
    parser=PageText(); parser.feed(raw.decode('utf-8','replace'))
    text=''.join(parser.parts); lines=[re.sub(r'\s+',' ',s).strip() for s in re.split(r'\n+|(?<=[.!?])\s+',text)]
    ranked=sorted(((sum(w in s.lower() for w in words),s) for s in lines if 40<=len(s)<=600),reverse=True)
    return [s[:350] for score,s in ranked if score>=2][:3]


def render_research(detections):
    from qc_web import search_qc_sources
    with st.expander('Online evidence retrieval (optional)'):
        st.caption('Search public pages for the detected class. Only short source excerpts are retrieved; untrusted text is not passed to the Llama prompt.')
        if st.button('Retrieve online evidence',key='final_web_evidence'):
            classes=sorted({ops.normalize(d['class']) for d in detections})
            words={'steel','surface','defect','inspection'}|{w for c in classes for w in c.split('_')}
            hits=[]
            with st.spinner('Retrieving source passages...'):
                try:
                    result=search_qc_sources(detections)
                    for s in result.get('sources',[])[:3]:
                        try: hits.append((s['title'],s['url'],page_extract(s['url'],words),''))
                        except Exception as e: hits.append((s['title'],s['url'],[],str(e)))
                except Exception as e: st.error(str(e))
            st.session_state['final_evidence']=(classes,hits)
        cached=st.session_state.get('final_evidence')
        if cached and cached[0]==sorted({ops.normalize(d['class']) for d in detections}):
            if not cached[1]: st.info('No pages available. Local guidance remains available.')
            for index,(title,url,passages,error) in enumerate(cached[1],1):
                st.write(f'[{index}] {title}'); st.link_button('Open source',url,key=f'final_evidence_{index}')
                for p in passages: st.text(p)
                if error: st.caption(error)
            st.caption('Retrieved excerpts are not verified corrective actions. Human QC approval required.')


def render_assistant(detections):
    from qc_llama import generate_grounded_qc_answer
    from llama_explain import format_qc_explanation
    with st.expander('Ask QC assistant'):
        st.caption('Offline assistant for known defect classes: checks, possible causes and corrective guidance. Not unrestricted question answering.')
        options=['Detected classes']+list(ops.GUIDANCE)
        choice=st.selectbox('Defect context',options,key='final_chat_class')
        question=st.text_input('Ask about checks, causes or corrective actions',key='final_chat_question')
        if st.button('Ask assistant',key='final_chat_ask'):
            ds=detections if choice=='Detected classes' else [{'class':choice,'confidence':0.0}]
            if not question.strip(): st.warning('Enter a question.'); return
            if not ds: st.info('Select a known defect class first.'); return
            # Route only to existing grounded adviser; free text never enters native runner.
            try:
                with st.spinner('Preparing grounded QC guidance...'):
                    report=generate_grounded_qc_answer(ds)
                st.session_state['final_chat_answer']=(question,choice,report['report_markdown'],format_qc_explanation(report),sorted({ops.normalize(d['class']) for d in detections}))
            except Exception as e: st.error(str(e))
        ans=st.session_state.get('final_chat_answer')
        if ans and ans[:2]==(question,choice) and ans[4]==sorted({ops.normalize(d['class']) for d in detections}): 
            q=question.lower()
            intent = ('cause','reason','root') if any(w in q for w in ('cause','why','reason','root')) else ('action','correct','prevent','alternative') if any(w in q for w in ('fix','action','correct','prevent','solution','avoid')) else ('check','inspect','verify','visual')
            chunks=re.split(r'(?m)(?=^#{1,6} )',ans[2])
            relevant=[p for p in chunks if any(w in p.split('\n',1)[0].lower() for w in intent)]
            if relevant: st.markdown('\n'.join(relevant))
            else: st.markdown(ans[2])
            st.write(ans[3]); st.caption('Draft local guidance; questions select relevant report sections, not an unrestricted Llama prompt.')
            if relevant and st.checkbox('Show complete local report',key='final_full_local'): st.markdown(ans[2])


def render_voice(detections):
    with st.expander('Offline voice guidance'):
        lang=st.session_state.get('steel_qc_language','English'); voice={'English':'en','தமிழ்':'ta','हिन्दी':'hi'}.get(lang,'en')
        if st.button('Prepare spoken guidance',key='final_voice'):
            heading,notice,checks,empty=ops.local_guidance(detections,lang)
            text=' '.join([heading]+[name+'. '+check for name,check in checks]+([empty] if not checks else [])+[notice])
            try:
                result=subprocess.run(['espeak-ng','-v',voice,'-s','145','--stdout','--stdin'],input=text.encode(),capture_output=True,timeout=15,check=True)
                if not result.stdout.startswith(b'RIFF'): raise ValueError('No WAV output')
                st.session_state['final_voice_wav']=(lang,text,result.stdout)
            except Exception as exc: st.warning('Voice unavailable: '+str(exc))
        record=st.session_state.get('final_voice_wav'); heading,notice,checks,empty=ops.local_guidance(detections,lang)
        current=' '.join([heading]+[name+'. '+check for name,check in checks]+([empty] if not checks else [])+[notice])
        if record and record[:2]==(lang,current): st.audio(record[2],format='audio/wav')


def render_extensions(frame,detections):
    render_novelty(frame); render_assistant(detections); render_research(detections); render_voice(detections)


def render_capture_review(last,key):
    if 'original_rgb' not in last: st.caption('Original frame unavailable; capture another frame.'); return
    if st.button('Collect frame for learning',key='final_collect_'+key):
        try:
            frame=cv2.cvtColor(last['original_rgb'],cv2.COLOR_RGB2BGR)
            sample=collect_image(frame,last['detections'],f'camera_{key}.png'); st.success('Sample stored: '+sample['status'])
        except Exception as exc: st.error(str(exc))


def render_queue():
    st.subheader('Training sample review')
    counts=statistics(); st.write(f"Pending: {counts['pending']} · Approved: {counts['approved']}")
    with closing(sqlite3.connect(LEARN_DB)) as c:
        c.row_factory=sqlite3.Row
        rows=[dict(r) for r in c.execute("SELECT sha,source_name,collected_at FROM samples WHERE status='pending' ORDER BY collected_at DESC LIMIT 300")]
        if not rows: st.info('No pending samples.'); return
        by_id={r['sha']:r for r in rows}; sha=st.selectbox('Choose sample',list(by_id),format_func=lambda k:f"{by_id[k]['source_name']} · {k[:10]}",key='final_queue_select')
        row=c.execute('SELECT image_png,predictions_json FROM samples WHERE sha=?',(sha,)).fetchone()
    frame=cv2.imdecode(np.frombuffer(row['image_png'],np.uint8),cv2.IMREAD_COLOR)
    if frame is None: st.error('Unreadable sample'); return
    ds=json.loads(row['predictions_json']); st.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),caption='Original sample',width=400)
    from dashboard_ui import numbered_annotations
    st.image(cv2.cvtColor(numbered_annotations(frame,ds),cv2.COLOR_BGR2RGB),caption='Predicted annotations',width=400)
    render_learning_feedback(frame,ds,by_id[sha]['source_name'],'queue'+sha)
STEEL_FINAL_0

cat > "$TMP/steel_i18n.py" <<'STEEL_FINAL_1'
"""Offline translations of operator-facing controls. Data values stay unchanged."""
import streamlit as st
# English | Tamil | Hindi. Unmapped technical text intentionally remains English.
RAW='''Inspection|ஆய்வு|निरीक्षण
Live camera|நேரடி கேமரா|लाइव कैमरा
Inspection history|ஆய்வு வரலாறு|निरीक्षण इतिहास
Model history|மாதிரி வரலாறு|मॉडल इतिहास
Batch inspection|தொகுப்பு ஆய்வு|बैच निरीक्षण
Alerts|எச்சரிக்கைகள்|अलर्ट
Power & recovery|மின்சாரம் மற்றும் மீட்பு|पावर और रिकवरी
Power and recovery|மின்சாரம் மற்றும் மீட்பு|पावर और रिकवरी
System health|சாதன நிலை|सिस्टम स्वास्थ्य
Inspection analytics|ஆய்வு பகுப்பாய்வு|निरीक्षण विश्लेषण
Inspection reports|ஆய்வு அறிக்கைகள்|निरीक्षण रिपोर्ट
Training sample review|பயிற்சி மாதிரி மதிப்பாய்வு|प्रशिक्षण नमूना समीक्षा
Settings|அமைப்புகள்|सेटिंग्स
Confidence threshold|நம்பிக்கை வரம்பு|विश्वास सीमा
Image size|பட அளவு|छवि आकार
Show YOLO bounding-box annotations|YOLO குறியீட்டுப் பெட்டிகளைக் காட்டு|YOLO बॉक्स दिखाएँ
Demo QC rules only. Confidence is not severity.|மாதிரி தர விதிகள் மட்டும். நம்பிக்கை மதிப்பு தீவிரம் அல்ல.|केवल प्रारूप गुणवत्ता नियम। विश्वास गंभीरता नहीं है।
AI-powered surface quality inspection|AI மேற்பரப்பு தர ஆய்வு|AI आधारित सतह गुणवत्ता निरीक्षण
Input source|உள்ளீட்டு மூலம்|इनपुट स्रोत
Image upload|பட பதிவேற்றம்|छवि अपलोड
Camera snapshot|கேமரா படம்|कैमरा स्नैपशॉट
Upload a steel surface image|எஃகு மேற்பரப்பு படத்தை பதிவேற்றவும்|स्टील की सतह की छवि अपलोड करें
Capture a still photo from your browser camera|கேமராவில் படம் எடுக்கவும்|कैमरे से फोटो लें
Choose a steel surface image to begin inspection.|ஆய்வைத் தொடங்க எஃகு மேற்பரப்பு படத்தைத் தேர்ந்தெடுக்கவும்.|निरीक्षण शुरू करने के लिए स्टील की सतह की छवि चुनें।
Inspection summary|ஆய்வு சுருக்கம்|निरीक्षण सारांश
Detected regions|கண்டறியப்பட்ட பகுதிகள்|पहचाने गए क्षेत्र
Inference time|கணிப்பு நேரம்|अनुमान समय
QC review|தர மதிப்பாய்வு|गुणवत्ता समीक्षा
Original image|அசல் படம்|मूल छवि
Annotated image|குறியிட்ட படம்|चिह्नित छवि
Annotated output|குறியிட்ட படம்|चिह्नित परिणाम
AI inspection guidance|AI ஆய்வு வழிகாட்டுதல்|AI निरीक्षण मार्गदर्शन
Generate QC guidance|தர ஆய்வு வழிகாட்டுதல் பெறுக|गुणवत्ता मार्गदर्शन प्राप्त करें
Generate defect-specific inspection guidance.|குறைபாட்டிற்கான ஆய்வு வழிகாட்டுதலை உருவாக்குக.|दोष के लिए निरीक्षण मार्गदर्शन प्राप्त करें।
Online technical references (optional)|இணைய தொழில்நுட்ப ஆதாரங்கள் (விருப்பம்)|ऑनलाइन तकनीकी संदर्भ (वैकल्पिक)
Search Web for More Solutions|இணையத்தில் மேலும் தேடுக|और समाधान खोजें
Download annotated PNG|குறியிட்ட PNG பதிவிறக்கம்|चिह्नित PNG डाउनलोड
Save inspection to SQLite|ஆய்வைச் சேமி|निरीक्षण सहेजें
Improve AI model - review annotations|AI மேம்பாடு - குறியீடுகளை மதிப்பாய்வு செய்க|AI सुधार - एनोटेशन की समीक्षा
Collected|சேகரித்தவை|एकत्रित
Approved|அங்கீகரித்தவை|स्वीकृत
Pending review|மதிப்பாய்வு நிலுவை|समीक्षा लंबित
Approve training sample|பயிற்சி மாதிரியை அங்கீகரி|प्रशिक्षण नमूना स्वीकार करें
Exclude from training|பயிற்சியிலிருந்து நீக்கு|प्रशिक्षण से हटाएँ
Live camera inspection|நேரடி கேமரா ஆய்வு|लाइव कैमरा निरीक्षण
Live video source|நேரடி காணொளி மூலம்|लाइव वीडियो स्रोत
USB webcam on Raspberry Pi|Pi USB கேமரா|Pi USB कैमरा
Raspberry Pi Camera Module (CSI)|Pi கேமரா தொகுதி (CSI)|Pi कैमरा मॉड्यूल (CSI)
RTSP / HTTP network camera|RTSP / HTTP வலை கேமரா|RTSP / HTTP नेटवर्क कैमरा
Browser webcam (Windows or phone) — WebRTC|உலாவி கேமரா (WebRTC)|ब्राउज़र कैमरा (WebRTC)
Start live detection|நேரடி ஆய்வைத் தொடங்கு|लाइव निरीक्षण शुरू करें
Stop and review latest frame|நிறுத்தி இறுதிப் படத்தை ஆய்வு செய்க|रोकें और अंतिम फ्रेम देखें
▶ Start live detection|நேரடி ஆய்வைத் தொடங்கு|लाइव निरीक्षण शुरू करें
■ Stop and review latest frame|நிறுத்தி இறுதிப் படத்தை ஆய்வு செய்க|रोकें और अंतिम फ्रेम देखें
Download live-frame annotations|குறியிட்ட கேமரா படம் பதிவிறக்கம்|चिह्नित कैमरा फ्रेम डाउनलोड
⬇️ Download live-frame annotations|குறியிட்ட கேமரா படம் பதிவிறக்கம்|चिह्नित कैमरा फ्रेम डाउनलोड
View saved image|சேமித்த படத்தைப் பார்|सहेजी गई छवि देखें
No saved inspections yet.|சேமித்த ஆய்வுகள் இல்லை.|अभी कोई सहेजा गया निरीक्षण नहीं है।
Export CSV|CSV ஏற்றுமதி|CSV निर्यात
Active version|செயலில் உள்ள பதிப்பு|सक्रिय संस्करण
Approved samples|அங்கீகரித்த மாதிரிகள்|स्वीकृत नमूने
Training status|பயிற்சி நிலை|प्रशिक्षण स्थिति
Waiting for data|தரவுக்காகக் காத்திருக்கிறது|डेटा की प्रतीक्षा
Restore previously active model|முந்தைய மாதிரியை மீட்டமை|पिछला मॉडल बहाल करें
I confirm the model rollback|மாதிரி மீட்டமைப்பை உறுதி செய்கிறேன்|मैं मॉडल वापसी की पुष्टि करता हूँ
Rollback model|முந்தைய மாதிரியை மீட்டமை|मॉडल वापस करें
Model decisions and evaluation history|மாதிரி முடிவுகள் மற்றும் மதிப்பீடு|मॉडल निर्णय और मूल्यांकन
Recent training jobs|சமீபத்திய பயிற்சிப் பணிகள்|हाल के प्रशिक्षण कार्य
No training jobs yet.|பயிற்சிப் பணிகள் இன்னும் இல்லை.|अभी प्रशिक्षण कार्य नहीं हैं।
Rollback becomes available after the first successful model replacement.|முதல் வெற்றிகரமான மாதிரி மாற்றத்திற்குப் பின் மீட்டமைக்கலாம்.|पहले सफल मॉडल बदलाव के बाद वापसी उपलब्ध होगी।
Choose steel surface images|எஃகு மேற்பரப்பு படங்களைத் தேர்ந்தெடு|स्टील की सतह की छवियाँ चुनें
Inspect and save batch|தொகுப்பை ஆய்வு செய்து சேமி|बैच का निरीक्षण और संग्रह करें
Download batch CSV|தொகுப்பு CSV பதிவிறக்கம்|बैच CSV डाउनलोड
QC alerts|தர எச்சரிக்கைகள்|गुणवत्ता अलर्ट
Unacknowledged alerts|உறுதிசெய்யாத எச்சரிக்கைகள்|अपुष्ट अलर्ट
Recent records|சமீபத்திய பதிவுகள்|हाल के रिकॉर्ड
Acknowledge|உறுதிசெய்|स्वीकार करें
Acknowledged|உறுதிசெய்தது|स्वीकृत
Recent alert history|சமீபத்திய எச்சரிக்கை வரலாறு|हाल का अलर्ट इतिहास
No alerts recorded yet.|எச்சரிக்கைகள் இல்லை.|अभी कोई अलर्ट दर्ज नहीं है।
Check local storage integrity|சேமிப்பக ஒருமைப்பாட்டைச் சரிபார்|स्थानीय संग्रह की अखंडता जाँचें
Reload dashboard|திரையை மீண்டும் ஏற்று|डैशबोर्ड पुनः लोड करें
Power action|மின்சார செயல்|पावर क्रिया
Shutdown|முடக்கு|बंद करें
Reboot|மீண்டும் தொடங்கு|पुनः प्रारंभ
Execute power action|மின்சார செயலைச் செய்|पावर क्रिया करें
I confirm that all inspection work is saved and camera streams are stopped.|ஆய்வுகள் சேமிக்கப்பட்டன; கேமராக்கள் நிறுத்தப்பட்டன என உறுதிசெய்கிறேன்.|मैं पुष्टि करता हूँ कि निरीक्षण सहेजे गए हैं और कैमरे बंद हैं।
CPU utilization|CPU பயன்பாடு|CPU उपयोग
Memory utilization|நினைவக பயன்பாடு|मेमोरी उपयोग
CPU temperature|CPU வெப்பநிலை|CPU तापमान
Free storage|காலி சேமிப்பகம்|खाली संग्रह
Saved inspections|சேமித்த ஆய்வுகள்|सहेजे गए निरीक्षण
High review|உயர் மதிப்பாய்வு|उच्च समीक्षा
Defect distribution|குறைபாட்டு விநியோகம்|दोष वितरण
Inspection activity by day|நாள் வாரியான ஆய்வுகள்|दिनवार निरीक्षण
Review categories|மதிப்பாய்வு வகைகள்|समीक्षा श्रेणियाँ
Select saved inspection|சேமித்த ஆய்வைத் தேர்ந்தெடு|सहेजा गया निरीक्षण चुनें
Prepare PDF report|PDF அறிக்கை தயார் செய்|PDF रिपोर्ट तैयार करें
Download PDF report|PDF அறிக்கை பதிவிறக்கம்|PDF रिपोर्ट डाउनलोड
Save an inspection first.|முதலில் ஆய்வைச் சேமிக்கவும்.|पहले निरीक्षण सहेजें।
Ask QC assistant|தர உதவியாளரிடம் கேள்|गुणवत्ता सहायक से पूछें
Defect context|குறைபாட்டு வகை|दोष संदर्भ
Detected classes|கண்டறிந்த வகைகள்|पहचानी गई श्रेणियाँ
Ask about checks, causes or corrective actions|ஆய்வு, காரணம் அல்லது திருத்தம் பற்றிக் கேள்|जाँच, कारण या सुधार के बारे में पूछें
Ask assistant|உதவியாளரிடம் கேள்|सहायक से पूछें
Online evidence retrieval (optional)|இணைய ஆதாரத் தேடல் (விருப்பம்)|ऑनलाइन साक्ष्य प्राप्ति (वैकल्पिक)
Retrieve online evidence|இணைய ஆதாரங்களைப் பெறு|ऑनलाइन साक्ष्य प्राप्त करें
Open source|ஆதாரத்தைத் திற|स्रोत खोलें
Offline voice guidance|இணையமில்லா குரல் வழிகாட்டுதல்|ऑफलाइन आवाज मार्गदर्शन
Prepare spoken guidance|குரல் வழிகாட்டுதல் தயார் செய்|आवाज मार्गदर्शन तैयार करें
Collect frame for learning|பயிற்சிக்காகப் படத்தைச் சேகரி|सीखने के लिए फ्रेम संग्रह करें
Choose sample|மாதிரியைத் தேர்ந்தெடு|नमूना चुनें
Original sample|அசல் மாதிரி|मूल नमूना
Predicted annotations|கணிக்கப்பட்ட குறியீடுகள்|अनुमानित एनोटेशन
No pending samples.|நிலுவை மாதிரிகள் இல்லை.|कोई लंबित नमूना नहीं है।
Research prototype only. NEU-DET lacks verified normal examples. No detections does not prove quality acceptance.|ஆராய்ச்சி மாதிரி மட்டும். NEU-DET இல் உறுதிசெய்த குறைபாடற்ற படங்கள் இல்லை. கண்டறிதல் இல்லாதது தர ஏற்பை உறுதிப்படுத்தாது.|केवल शोध प्रोटोटाइप। NEU-DET में सत्यापित सामान्य उदाहरण नहीं हैं। दोष न मिलना गुणवत्ता स्वीकृति का प्रमाण नहीं है।'''
TRANSLATIONS={}
for row in RAW.splitlines():
    en,ta,hi=row.split('|'); TRANSLATIONS[en]=(ta,hi)

def tr(text):
    if not isinstance(text,str): return text
    lang=st.session_state.get('steel_qc_language','English')
    if lang=='English' or text not in TRANSLATIONS: return text
    return TRANSLATIONS[text][0 if lang=='தமிழ்' else 1]

def language_selector():
    st.sidebar.selectbox('Language / மொழி / भाषा',['English','தமிழ்','हिन्दी'],key='steel_qc_language')
    st.sidebar.caption(tr('Settings')+' · English technical data retained')

def apply_style():
    st.markdown('''<style>
    .block-container {padding-top: 2.4rem; max-width: 1450px;}
    [data-testid="stTabs"] [role="tablist"] {gap: .4rem; padding: .35rem 0 .65rem; flex-wrap: wrap; overflow: visible;}
    [data-testid="stTabs"] [role="tab"] {border:1px solid #475569; border-radius:8px; background:#202936; min-height:40px; padding:0 .85rem; color:#e2e8f0;}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {background:#1d4ed8; border-color:#60a5fa; color:white;}
    [data-testid="stTabs"] [role="tab"] p {color:inherit; font-weight:600;}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {background:transparent!important;}
    [data-testid="stTabs"] [data-baseweb="tab-border"] {background:transparent!important;}
    button[kind="primary"]:not(:disabled), button[data-testid="stBaseButton-primary"]:not(:disabled) {background:#1d4ed8!important; color:#fff!important; border-color:#60a5fa!important;}
    button[kind="primary"]:not(:disabled) p, button[data-testid="stBaseButton-primary"]:not(:disabled) p {color:#fff!important;}
    </style>''',unsafe_allow_html=True)
STEEL_FINAL_1

cat > "$TMP/patch_final.py" <<'STEEL_FINAL_2'
import ast,sys,json,shutil
from pathlib import Path
root,package,backup=map(Path,sys.argv[1:4]); dash=root/'src/dashboard'

def apply_edits(s, edits):
    lines=s.encode().splitlines(keepends=True); starts=[0]
    for line in lines: starts.append(starts[-1]+len(line))
    def pos(line,col): return starts[line-1]+col
    data=s.encode()
    for a,b,v in sorted([(pos(*a),pos(*b),v.encode()) for a,b,v in edits],reverse=True): data=data[:a]+v+data[b:]
    return data.decode()
def span(n): return (n.lineno,n.col_offset),(n.end_lineno,n.end_col_offset)
def callname(n): return n.func.attr if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) else ''
def is_st(n):
    if not isinstance(n,ast.Call) or not isinstance(n.func,ast.Attribute): return False
    v=n.func.value
    while isinstance(v,ast.Attribute): v=v.value
    return isinstance(v,ast.Name) and v.id=='st' or isinstance(v,ast.Name) and v.id in ('a','b','c','d','start_col','stop_col')
def bykey(n,key): return isinstance(n,ast.Call) and any(k.arg=='key' and isinstance(k.value,ast.Constant) and k.value.value==key for k in n.keywords)
def insertion_after(s,n,code):
    lines=s.splitlines(keepends=True); lines.insert(n.end_lineno,' '*n.col_offset+code+'\n'); return ''.join(lines)
def import_after_future(s,line):
    t=ast.parse(s); last=0
    if t.body and isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str): last=t.body[0].end_lineno
    for n in t.body:
        if isinstance(n,ast.ImportFrom) and n.module=='__future__': last=n.end_lineno
    lines=s.splitlines(keepends=True); lines.insert(last,line+'\n'); return ''.join(lines)

# Import translator in dashboard modules only (no changes to detector classes or storage).
import importlib.util
spec=importlib.util.spec_from_file_location('dictionary',package/'steel_i18n.py')
# Avoid loading streamlit/session state merely to read the translation literals.
raw=next(n.value.value for n in ast.parse((package/'steel_i18n.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RAW' for t in n.targets))
known={r.split('|')[0] for r in raw.splitlines()}
methods={'title','header','subheader','caption','write','markdown','info','warning','error','success','button','download_button','file_uploader','camera_input','checkbox','radio','selectbox','select_slider','slider','text_input','number_input','expander','form_submit_button','metric','spinner','link_button','image'}
def localize(s):
    edits=[]
    for n in ast.walk(ast.parse(s)):
        if not isinstance(n,ast.Call) or not is_st(n): continue
        name=callname(n)
        if name=='tabs' and n.args and isinstance(n.args[0],(ast.List,ast.Tuple)):
            for v in n.args[0].elts:
                if isinstance(v,ast.Constant) and isinstance(v.value,str) and v.value in known: edits.append((*span(v),'tr('+repr(v.value)+')'))
        if name not in methods: continue
        if n.args and name!='image':
            v=n.args[0]
            if isinstance(v,ast.Constant) and isinstance(v.value,str) and v.value in known: edits.append((*span(v),'tr('+repr(v.value)+')'))
        for k in n.keywords:
            v=k.value
            if k.arg in ('label','caption','help','placeholder') and isinstance(v,ast.Constant) and isinstance(v.value,str) and v.value in known: edits.append((*span(v),'tr('+repr(v.value)+')'))
        if name in ('radio','selectbox') and len(n.args)>1 and isinstance(n.args[1],(ast.List,ast.Tuple)) and any(isinstance(v,ast.Constant) and v.value in known for v in n.args[1].elts) and not any(k.arg=='format_func' for k in n.keywords):
            # Keep original option values; only translate displayed options.
            edits.append(((n.end_lineno,n.end_col_offset-1),(n.end_lineno,n.end_col_offset-1),', format_func=tr' if not ast.get_source_segment(s,n).rstrip()[:-1].rstrip().endswith(',') else ' format_func=tr'))
    return import_after_future(apply_edits(s,edits),'from steel_i18n import tr')

app=dash/'app_pi.py'; s=app.read_text()
assert 'from operator_ui import render_batch' in s and 'render_model_history' in s and 'render_learning_feedback' in s, 'Install operator and automatic learning add-ons first'
assert '# Steel final features' not in s, 'Final installer already applied'
tree=ast.parse(s)
selector=[n for n in tree.body if isinstance(n,ast.Expr) and bykey(n.value,'steel_qc_language')]; assert len(selector)==1
page=[n for n in tree.body if isinstance(n,ast.Expr) and callname(n.value)=='set_page_config']; assert len(page)==1
nav=[n for n in tree.body if isinstance(n,ast.Assign) and callname(n.value)=='tabs']; assert len(nav)==1
n=nav[0]; assert isinstance(n.targets[0],ast.Tuple) and isinstance(n.value.args[0],ast.List)
labels=[ast.literal_eval(v) for v in n.value.args[0].elts]; targets=[v.id for v in n.targets[0].elts]; assert len(labels)==7
new_labels=['System health','Inspection analytics','Inspection reports','Training sample review']; new_targets=['health_tab','analytics_tab','reports_tab','review_queue_tab']
lines=s.splitlines(keepends=True)
for start,end,rep in sorted([(selector[0].lineno,selector[0].end_lineno,''),(page[0].end_lineno+1,page[0].end_lineno,'from steel_i18n import language_selector, apply_style\nlanguage_selector()\napply_style()\n'),(n.lineno,n.end_lineno,', '.join(targets+new_targets)+' = st.tabs('+repr(labels+new_labels)+')\n')],reverse=True): lines[start-1:end]=[rep]
s=''.join(lines)
# Add extension panel below annotation review, using exact existing locals.
tree=ast.parse(s); calls=[n for n in ast.walk(tree) if isinstance(n,ast.Expr) and callname(n.value)=='render_learning_feedback']; assert len(calls)==1
s=insertion_after(s,calls[0],'render_extensions(frame, detections)')
# Captured Pi frame: explicit collection; no continuous camera storage.
tree=ast.parse(s); calls=[n for n in ast.walk(tree) if isinstance(n,ast.Expr) and bykey(n.value,'pi_live_download')]; assert len(calls)==1
s=insertion_after(s,calls[0],"render_capture_review(last, 'pi')")
s=import_after_future(s,'# Steel final features\nfrom final_features import render_health, render_analytics, render_reports, render_queue, render_extensions, render_capture_review')
# imports must follow src/inference path setup and learning imports. Move extension import after known operator import instead.
imp='from final_features import render_health, render_analytics, render_reports, render_queue, render_extensions, render_capture_review\n'; s=s.replace(imp,'',1)
anchor='from learning_feedback import render_learning_feedback'; assert s.count(anchor)==1; s=s.replace(anchor,anchor+'\n'+imp.rstrip(),1)
s+='\n\nwith health_tab:\n    render_health()\nwith analytics_tab:\n    render_analytics(DB)\nwith reports_tab:\n    render_reports(DB, OUT)\nwith review_queue_tab:\n    render_queue()\n'
files={app:localize(s)}
browser=dash/'browser_live.py'; s=browser.read_text(); t=ast.parse(s)
records=[n for n in ast.walk(t) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Dict) and any(isinstance(t,ast.Attribute) and t.attr=='_last' for t in n.targets)]; assert len(records)==1
r=records[0].value
if not any(isinstance(k,ast.Constant) and k.value=='original_rgb' for k in r.keys): s=apply_edits(s,[((r.lineno,r.col_offset+1),(r.lineno,r.col_offset+1),'"original_rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),')])
t=ast.parse(s); candidates=[n for n in ast.walk(t) if isinstance(n,ast.Expr) and callname(n.value)=='download_button']; assert len(candidates)==1
n=candidates[0]; lines=s.splitlines(keepends=True); indent=' '*n.col_offset; lines.insert(n.end_lineno,indent+"if not context.state.playing:\n"+indent+"    from final_features import render_capture_review\n"+indent+"    render_capture_review(last, 'browser')\n"); s=''.join(lines); files[browser]=localize(s)
for name in ['dashboard_ui.py','learning_feedback.py','model_history.py','operator_ui.py']:
    p=dash/name; files[p]=localize(p.read_text())
for name in ['final_features.py','steel_i18n.py']:
    p=dash/name; files[p]=(package/name).read_text() if name=='steel_i18n.py' else localize((package/name).read_text())
for p,s in files.items(): compile(s,str(p),'exec')
manifest={str(p.relative_to(root)):p.exists() for p in files}
for p in files:
    if p.exists(): q=backup/p.relative_to(root); q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
(backup/'manifest.json').write_text(json.dumps(manifest,indent=2))
for p,s in files.items(): p.write_text(s); print('Updated',p.relative_to(root))
print('Source backups:',backup)
STEEL_FINAL_2

cat > "$TMP/FINAL_CHECKLIST.md" <<'STEEL_FINAL_3'
# Steel Inspector | Release checklist

## Deployment
All AI execution, CPU training, database operations, model selection, and fullscreen display run on the Pi 5. Local service: `steel-inspector.service`; learning scheduler: `steel-learning.timer`; local URL: `http://127.0.0.1:8501`. Online evidence search is optional and requires connectivity. Do not describe it as offline.

## Architecture
```mermaid
flowchart TD
 A[Pi USB or CSI camera / local image] --> B[OpenCV + YOLO NCNN]
 B --> C[Inspection dashboard / SQLite / annotated evidence]
 B --> D[Local draft QC knowledge + ExecuTorch Llama]
 B --> E[Human-approved annotations]
 E --> F[Versioned train dataset + original train samples]
 F --> G[CPU training on Pi]
 G --> H[NCNN export and fixed validation comparison]
 H --> I{Promotion gates pass?}
 I -->|Yes| J[Atomic model registry activation]
 I -->|No| K[Retain previous model]
 J --> B
 C --> L[PDF, analytics, alerts]
```

## Final validation (record actual outcomes, not assumed passes)
- Confirm cold boot directly to the display, upload, offline Llama, annotation visibility and SQLite save. Disconnect external network for core offline verification.
- Exercise USB/CSI cameras actually available; collect a stopped frame and approve corrected annotations in Training sample review. Verify browser snapshots separately.
- Batch inspect, confirm saved records, CSV and alert acknowledgement. Confirm PDF downloads and image/report matching. Verify Tamil/Hindi controls and audio on local display/speakers.
- Check dashboard health and inspect temperature/throttling. Use a stable Pi 5 PSU and active cooling for training. Measure electrical watts with an external meter; CPU load is not watts.
- Evaluate on untouched test set only after model selection. Record class-wise precision/recall, mAP, F1 and repeated warm latency, hardware/RAM/temp and timing method. Do not use test data for training or promotion decisions. NEU-DET does not establish normal-surface acceptance.
- Collect 50 truly new, distinct, reviewed samples (do not approve original validation/test examples). Monitor systemd journal and training status; record whether candidate is rejected or promoted. If promoted verify active version changes and previous model can be restored. Never force a promotion just to demonstrate a success.
- Demonstrate unfamiliar-texture warning as experimental OOD screening; no claim of unknown-defect accuracy without independent evaluation.
- Demonstrate source excerpts with attribution and offline fallback. Local QC assistant is restricted to grounded known-class guidance; online excerpts are not approved process instructions.
- Safe shutdown only when inspection/training idle. Reboot and confirm data/model history persist.
- Run `bash docs/collect_final_evidence.sh` and retain logs. Review failures before release.

## Submission handoff
- Review `git status`, `git diff` and repository README. Stage reviewed source, docs, tests, dependency manifests and necessary configuration only; exclude `.bak`, `.venv`, datasets, weights, database files, secrets, logs and generated evidence. Never run indiscriminate `git add .`.
- Update technical report (8–10 pages): problem, design, methodology, dataset split, offline inference, training mechanism, optimization, measured results, limitations, conclusion. Include actual Pi display/screenshots and architecture. Label pending results as pending.
- PPT suggested sequence: problem, solution, architecture, edge optimization, live workflow, model learning, observed results, limitations and next steps.
- Demo video sequence: cold boot → image or camera → defect boxes → offline QC → save → batch/alerts → health/analytics/PDF → model history and training eligibility → power control. If no candidate has passed, show waiting or rejected status; do not stage a fake promotion.
- Verify challenge submission form and requirements independently. Commit and push only after reviewing content and credentials.
STEEL_FINAL_3

cat > "$TMP/finish_docs.sh" <<'STEEL_FINAL_4'
#!/usr/bin/env bash
# Read-only evidence collection. Does not start training or turn off the Pi.
set -euo pipefail
cd "$HOME/Industrial_Defect_Detection"
PY="$HOME/soc-venvs/cpu-validation-BdkS2KBp/.venv/bin/python"
OUT="$HOME/soc-retraining/evidence/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
{
 date -Is; uname -a; cat /proc/device-tree/model; echo
 free -h; df -h "$HOME"; vcgencmd measure_temp; vcgencmd get_throttled
 systemctl is-active steel-inspector.service || true
 systemctl list-timers steel-learning.timer --all --no-pager
 cat "$HOME/soc-retraining/training_status.json"
 git status --short; git diff --stat
 "$PY" -m pip check
} > "$OUT/system.txt" 2>&1
"$PY" -m unittest discover -s tests -v > "$OUT/regression.txt" 2>&1 || echo 'Regression failure: inspect regression.txt'
PYTHONPATH=src "$PY" - <<'PY' > "$OUT/model.txt"
from learning import model_registry as reg
from learning.feedback_store import statistics
import json
s=reg.read_state(); reg.verify_entry(s['versions'][s['active']]); print('Active model integrity OK'); print(json.dumps(s,indent=2)); print(statistics())
PY
printf 'Evidence: %s\n' "$OUT"
STEEL_FINAL_4
# Preflight and one-time optional setup. No dependencies fetched by runtime.
"$PY" -m py_compile "$TMP"/*.py
"$PY" -c 'import streamlit, cv2, numpy, pandas, psutil'
if ! "$PY" -c 'import reportlab' >/dev/null 2>&1; then
  "$PY" -m pip install --timeout 20 --retries 1 reportlab || echo 'PDF dependency missing: PDF tab will report unavailable until installed.'
fi
if ! command -v espeak-ng >/dev/null 2>&1; then
  sudo apt-get install -y espeak-ng || echo 'Voice dependency unavailable; other features continue.'
fi
if command -v fc-list >/dev/null && ! fc-list :lang=ta | grep -q .; then
  sudo apt-get install -y fonts-noto-core || echo 'Tamil/Hindi fonts may require manual installation.'
fi
"$PY" -m pip check
sudo systemctl stop steel-learning.timer
TIMER_STOPPED=1
state=$(systemctl show -p ActiveState --value steel-learning.service)
case "$state" in inactive|failed) ;; *) echo 'Training is active. No files changed. Retry when idle.'; exit 1;; esac
sudo systemctl stop steel-inspector.service
STOPPED=1
cd "$ROOT"
"$PY" "$TMP/patch_final.py" "$ROOT" "$TMP" "$BACKUP"
# Build local reference from restored train/val, never the held-out test data.
PYTHONPATH="$ROOT/src:$ROOT/src/dashboard:$ROOT/src/inference" "$PY" - <<'BUILD'
from final_features import build_reference
try:
 print('Texture reference (train, val, threshold):',build_reference())
except Exception as e:
 print('Texture reference unavailable:',e)
BUILD
mkdir -p docs
install -m 644 "$TMP/FINAL_CHECKLIST.md" docs/FINAL_RELEASE_CHECKLIST.md
install -m 755 "$TMP/finish_docs.sh" docs/collect_final_evidence.sh
# Compile every modified dashboard source, then health check after restart.
"$PY" -m compileall -q src/dashboard
sudo systemctl start steel-inspector.service
READY=0
for i in $(seq 1 30); do
 if curl -fsS --max-time 2 http://127.0.0.1:8501/_stcore/health >/dev/null; then READY=1; break; fi
 sleep 2
done
[ "$READY" = 1 ] || { echo 'Dashboard health endpoint failed'; journalctl -u steel-inspector.service -n 30 --no-pager; exit 1; }
COMMITTED=1; STOPPED=0
sudo systemctl start steel-learning.timer
TIMER_STOPPED=0
printf '\nFINAL FEATURE UPDATE INSTALLED\nSource backup: %s\n' "$BACKUP"
echo 'Refresh the dashboard. Browser rendering and actual hardware behavior require verification.'
echo 'Release checklist: docs/FINAL_RELEASE_CHECKLIST.md'
echo 'Collect evidence later: bash docs/collect_final_evidence.sh'
systemctl is-active steel-inspector.service
systemctl list-timers steel-learning.timer --all --no-pager
