from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
from common import DB_PATH, init_db

st.set_page_config(page_title="Window AC Dent Inspection", page_icon="🔎", layout="wide")
init_db()
st.title("AI Visual Inspection — Window AC Condenser")
st.caption("Local V1 dashboard | GOOD vs DENT | SQLite event log")

if st.button("Refresh data"):
    st.rerun()

if not DB_PATH.exists():
    st.info("No database yet. Run live_inference.py after training a model.")
    st.stop()

with sqlite3.connect(DB_PATH) as con:
    df = pd.read_sql_query(
        "SELECT id,timestamp,prediction,confidence,status,image_path,source FROM inspections ORDER BY timestamp DESC",
        con
    )

if df.empty:
    st.info("No inspections logged yet. Start live_inference.py in a separate terminal.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["hour"] = df["timestamp"].dt.floor("h")
total = len(df)
good = int((df["status"] == "GOOD").sum())
dent = int(df["status"].isin(["DENT", "NG"]).sum())
review = int((df["status"] == "REVIEW").sum())
ng_rate = (dent / total * 100) if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total inspection samples", f"{total:,}")
c2.metric("GOOD", f"{good:,}")
c3.metric("DENT / NG", f"{dent:,}")
c4.metric("DENT / NG rate", f"{ng_rate:.2f}%")

st.caption(f"REVIEW (below confidence threshold): {review:,}. Counts are prediction samples, not guaranteed unique products.")
hourly = df.groupby("hour", dropna=True).agg(
    inspections=("id", "count"),
    good=("status", lambda s: int((s == "GOOD").sum())),
    dent=("status", lambda s: int(s.isin(["DENT", "NG"]).sum())),
    review=("status", lambda s: int((s == "REVIEW").sum()))
).reset_index()
hourly["ng_rate_pct"] = hourly.apply(
    lambda r: (r["dent"] / r["inspections"] * 100) if r["inspections"] else 0, axis=1
)

left, right = st.columns(2)
with left:
    st.subheader("Inspection samples per hour")
    fig = px.bar(hourly, x="hour", y=["good", "dent", "review"], barmode="stack",
                 labels={"value": "Samples", "hour": "Hour", "variable": "Result"})
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Hourly DENT / NG rate")
    fig2 = px.line(hourly, x="hour", y="ng_rate_pct", markers=True,
                   labels={"ng_rate_pct": "NG rate (%)", "hour": "Hour"})
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Inspection history")
show = df.copy()
show["timestamp"] = show["timestamp"].astype(str)
st.dataframe(show[["id", "timestamp", "prediction", "confidence", "status", "source", "image_path"]],
             use_container_width=True, hide_index=True)

csv = df.to_csv(index=False).encode("utf-8-sig")
st.download_button("Download inspection log (CSV)", data=csv,
                   file_name="window_ac_dent_inspections.csv", mime="text/csv")

st.subheader("Latest saved image evidence")
selected = st.selectbox("Choose inspection ID", df["id"].astype(int).tolist())
row = df[df["id"] == selected].iloc[0]
img_path = Path(str(row["image_path"])) if pd.notna(row["image_path"]) else None
if img_path and img_path.exists():
    st.image(str(img_path), caption=f"ID {selected} | {row['status']} | confidence {float(row['confidence']):.1%}",
             use_container_width=True)
else:
    st.caption("Image evidence path is missing or file is not available.")
