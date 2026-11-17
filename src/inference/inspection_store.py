"""Local inspection persistence shared by uploaded and live-camera frames.

No model inference or internet access. Uses the existing SQLite schema.
"""

import csv
import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """CREATE TABLE IF NOT EXISTS pi_inspections (
    id TEXT PRIMARY KEY, timestamp_utc TEXT, filename TEXT, model TEXT,
    conf REAL, imgsz INTEGER, detection_count INTEGER,
    priority TEXT, recommendation TEXT, detections_json TEXT,
    annotated_file TEXT)"""


def init_db(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)


def save_inspection(db_path, output_dir, *, filename, model, conf, imgsz,
                    detections, priority, recommendation, annotated_png):
    """Save one annotated PNG and one inspection row; roll back file on DB error."""
    if not annotated_png or not isinstance(annotated_png, bytes):
        raise ValueError("Annotated image must be nonempty PNG bytes")
    init_db(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection_id = uuid.uuid4().hex[:12]
    location = output_dir / (inspection_id + ".png")
    location.write_bytes(annotated_png)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO pi_inspections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    inspection_id, datetime.now(timezone.utc).isoformat(),
                    Path(filename).name, str(model), float(conf), int(imgsz),
                    len(detections), str(priority), str(recommendation),
                    json.dumps(detections), str(location),
                ),
            )
    except Exception:
        location.unlink(missing_ok=True)
        raise
    return inspection_id


def load_inspections(db_path, limit=300):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [
            dict(row) for row in conn.execute(
                "SELECT * FROM pi_inspections "
                "ORDER BY timestamp_utc DESC LIMIT ?", (int(limit),)
            )
        ]


def export_inspections_csv(rows):
    columns = [
        "timestamp_utc", "filename", "detection_count",
        "priority", "recommendation",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return output.getvalue()
