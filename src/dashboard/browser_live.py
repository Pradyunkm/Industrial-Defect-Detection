"""Optional browser webcam live YOLO using streamlit-webrtc.

No browser-camera library is imported until user chooses this option.
Requires a browser secure context: HTTPS or localhost on the user's laptop.
"""
import json
import sqlite3
import threading
import numpy as np
import time
import uuid
from datetime import datetime, timezone

import cv2
import streamlit as st

from common import predict, review
from llama_explain import explain_defects


def render_browser_live(model, conf, imgsz, show_annotations, db_path, output_dir, model_path, fallback_guidance):
    st.subheader("Browser webcam — live YOLO")
    st.info(
        "Uses the camera of the device displaying this website (for example "
        "your Windows laptop webcam). Browser camera access requires HTTPS or "
        "a localhost URL. For local development, SSH port forwarding to the Pi "
        "and http://localhost:8501 can provide a secure browser context."
    )
    st.caption(
        "Browser webcam support is optional. It does not modify the Windows camera app. "
        "Detection runs on the Raspberry Pi CPU at a reduced update rate. "
        "Stop the browser video player before requesting a Llama explanation of a captured frame."
    )
    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    except ImportError:
        st.warning(
            "Browser live video requires additional packages in the Pi virtual environment: "
            "streamlit-webrtc and av. Existing USB, CSI, IP, upload and snapshot options "
            "continue working without these packages."
        )
        st.code("python -m pip install streamlit-webrtc", language="bash")
        return

    class BrowserYOLOProcessor(VideoProcessorBase):
        def __init__(self):
            self._lock = threading.Lock()
            self._processed_at = 0.0
            self._last = None
            self._error = None
            self.show_annotations = show_annotations

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            # Keep resource use controlled on the Pi: max 640 px width and
            # at most one YOLO prediction every two seconds.
            h, w = image.shape[:2]
            if w > 640:
                image = cv2.resize(image, (640, max(1, int(h * 640 / w))))
            now = time.monotonic()
            with self._lock:
                should_run = now - self._processed_at >= 2.0
                if should_run:
                    self._processed_at = now

            if should_run:
                try:
                    annotated, detections, elapsed_ms = predict(
                        image, model, conf=conf, imgsz=imgsz
                    )
                    priority, advice = review(detections)
                    ok, encoded = cv2.imencode(".png", annotated)
                    if ok:
                        with self._lock:
                            self._last = {
                                "annotated_png": encoded.tobytes(),
                                "detections": detections,
                                "elapsed_ms": elapsed_ms,
                                "priority": priority,
                                "advice": advice,
                                "model": model_path,
                                "conf": conf,
                                "imgsz": imgsz,
                            }
                            self._error = None
                    display = annotated if self.show_annotations else image
                except Exception as error:
                    with self._lock:
                        self._error = str(error)
                    display = image
            else:
                with self._lock:
                    snapshot = self._last
                if self.show_annotations and snapshot:
                    display = cv2.imdecode(
                        np.frombuffer(snapshot["annotated_png"], dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if display is None:
                        display = image
                else:
                    display = image
            return av.VideoFrame.from_ndarray(display, format="bgr24")

        def snapshot(self):
            with self._lock:
                return dict(self._last) if self._last else None, self._error

    context = webrtc_streamer(
        key="pi-browser-live-yolo",
        video_processor_factory=BrowserYOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": []},
        async_processing=True,
    )
    if context.video_processor:
        context.video_processor.show_annotations = show_annotations

    @st.fragment(run_every="3s")
    def browser_results():
        processor = context.video_processor
        if processor is not None:
            record, error = processor.snapshot()
            if error:
                st.error(f"Browser video inference: {error}")
            if record is not None:
                st.session_state["pi_browser_last"] = record
                st.session_state["pi_browser_explanation"] = None if (
                    st.session_state.get("pi_browser_last_png") != record["annotated_png"]
                ) else st.session_state.get("pi_browser_explanation")
                st.session_state["pi_browser_last_png"] = record["annotated_png"]

        last = st.session_state.get("pi_browser_last")
        if last is None:
            st.caption("Click START on the browser video player. Once YOLO has processed a frame, results appear here.")
            return

        st.write(
            f"**Last frame:** {len(last['detections'])} detections | "
            f"{last['elapsed_ms']:.0f} ms YOLO | **Review:** {last['priority']}"
        )
        st.caption(last["advice"])
        if last["detections"]:
            st.dataframe(last["detections"], hide_index=True, width="stretch")
        st.download_button(
            "⬇️ Download browser webcam annotated PNG",
            last["annotated_png"],
            "browser_webcam_annotated.png",
            mime="image/png",
            key="pi_browser_download",
        )

        if st.button("🤖 Explain browser webcam frame with Llama", key="pi_browser_llama"):
            try:
                with st.spinner("Generating offline inspection guidance..."):
                    report = explain_defects(last["detections"])
                    explanation = report["llama_explanation"].strip()
                    if len(explanation) < 60 or not explanation.endswith((".", "!", "?")):
                        explanation = fallback_guidance(last["detections"], report["summary"])
                    st.session_state["pi_browser_explanation"] = (report["summary"], explanation)
            except Exception as error:
                st.error(f"Llama explanation unavailable: {error}")

        if st.session_state.get("pi_browser_explanation"):
            summary, explanation = st.session_state["pi_browser_explanation"]
            st.write("**YOLO summary:**", summary)
            st.write("**Inspection guidance:**", explanation)
            st.caption("Advisory only. Human verification required.")

        if st.button("💾 Save browser webcam frame to SQLite", key="pi_browser_save"):
            record_id = uuid.uuid4().hex[:12]
            output_path = output_dir / (record_id + ".png")
            output_path.write_bytes(last["annotated_png"])
            try:
                with sqlite3.connect(db_path) as connection:
                    connection.execute(
                        "INSERT INTO pi_inspections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            record_id, datetime.now(timezone.utc).isoformat(),
                            "browser_webcam_capture.png", last["model"], last["conf"],
                            last["imgsz"], len(last["detections"]),
                            last["priority"], last["advice"],
                            json.dumps(last["detections"]), str(output_path)
                        )
                    )
            except Exception:
                output_path.unlink(missing_ok=True)
                raise
            st.success(f"Browser webcam inspection {record_id} saved.")

    browser_results()
