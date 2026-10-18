import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/inspection.db")
SNAPSHOT_DIR = Path("logs/snapshots")

# columns added in V2 (older databases are migrated automatically, old rows are kept)
V2_COLUMNS = {"det_score": "REAL", "box": "TEXT", "votes": "INTEGER"}


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            image_path TEXT,
            source TEXT NOT NULL DEFAULT 'webcam'
        )
        """)
        existing = {row[1] for row in con.execute("PRAGMA table_info(inspections)")}
        for col, typ in V2_COLUMNS.items():
            if col not in existing:
                con.execute(f"ALTER TABLE inspections ADD COLUMN {col} {typ}")
        con.execute("CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp)")


def log_inspection(prediction, confidence, status, image_path=None, source="webcam",
                   det_score=None, box=None, votes=None):
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO inspections(timestamp,prediction,confidence,status,image_path,source,det_score,box,votes) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),
             prediction, float(confidence), status, str(image_path) if image_path else None, source,
             None if det_score is None else float(det_score),
             None if box is None else ",".join(str(int(v)) for v in box),
             votes)
        )


def counts_today(source="webcam_unit"):
    """OK / NG / REVIEW totals logged today (local time) - used by live_inference --resume."""
    init_db()
    today = datetime.now().astimezone().date().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT status, COUNT(*) FROM inspections WHERE source=? AND substr(timestamp,1,10)=? GROUP BY status",
            (source, today)).fetchall()
    return {status: n for status, n in rows}


def fetch_records():
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        return con.execute(
            "SELECT id,timestamp,prediction,confidence,status,image_path,source "
            "FROM inspections ORDER BY timestamp DESC"
        ).fetchall()
