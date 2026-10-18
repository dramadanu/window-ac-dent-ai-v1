import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/inspection.db")
SNAPSHOT_DIR = Path("logs/snapshots")

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
        con.execute("CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp)")

def log_inspection(prediction, confidence, status, image_path=None, source="webcam"):
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO inspections(timestamp,prediction,confidence,status,image_path,source) "
            "VALUES(?,?,?,?,?,?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"),
             prediction, float(confidence), status, str(image_path) if image_path else None, source)
        )

def fetch_records():
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        return con.execute(
            "SELECT id,timestamp,prediction,confidence,status,image_path,source "
            "FROM inspections ORDER BY timestamp DESC"
        ).fetchall()
