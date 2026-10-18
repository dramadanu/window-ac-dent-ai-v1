"""Pandas helpers for the dashboard (kept separate so they can be tested without Streamlit)."""
import sqlite3

import pandas as pd

from common import DB_PATH, init_db

RESULT_MAP = {"GOOD": "OK", "OK": "OK", "DENT": "NG", "NG": "NG", "REVIEW": "REVIEW"}


def load_inspections() -> pd.DataFrame:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("SELECT * FROM inspections ORDER BY timestamp DESC", con)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df["hour"] = df["timestamp"].dt.floor("h")
    df["result"] = df["status"].str.upper().map(RESULT_MAP).fillna("REVIEW")
    return df


def summarize(df: pd.DataFrame) -> dict:
    total = len(df)
    n = {k: int((df["result"] == k).sum()) for k in ("OK", "NG", "REVIEW")}
    pct = {k: (100.0 * v / total if total else 0.0) for k, v in n.items()}
    return {"total": total, "n": n, "pct": pct}


def per_hour(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["hour", "result"]).size().unstack(fill_value=0)
    for k in ("OK", "NG", "REVIEW"):
        if k not in t:
            t[k] = 0
    t = t[["OK", "NG", "REVIEW"]]
    t["total"] = t.sum(axis=1)
    t["ng_rate_pct"] = 100.0 * t["NG"] / t["total"]
    return t.reset_index()
