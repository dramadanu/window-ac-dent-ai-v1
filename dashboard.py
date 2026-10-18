from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from common import DB_PATH, init_db
from stats import load_inspections, summarize, per_hour

st.set_page_config(page_title="Window AC Dent Inspection", page_icon="🔎", layout="wide")
init_db()
st.title("AI Visual Inspection — Window AC Condenser")
st.caption("V2 | condenser detection (bounding box) -> OK / NG classification | one record per unit")

if st.button("Refresh data"):
    st.rerun()

df_all = load_inspections()
if df_all.empty:
    st.info("No inspections logged yet. Run live_inference.py first.")
    st.stop()

mode = st.radio("Data", ["Units (V2 live inspection)", "Legacy frame samples (V1)"], horizontal=True)
df = df_all[df_all["source"] == "webcam_unit"] if mode.startswith("Units") else df_all[df_all["source"] != "webcam_unit"]
if df.empty:
    st.info("No records for this selection yet.")
    st.stop()

s = summarize(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total inspected", f"{s['total']:,}")
c2.metric("OK", f"{s['n']['OK']:,}", f"{s['pct']['OK']:.1f}%", delta_color="off")
c3.metric("NG (dent)", f"{s['n']['NG']:,}", f"{s['pct']['NG']:.1f}%", delta_color="off")
c4.metric("REVIEW", f"{s['n']['REVIEW']:,}", f"{s['pct']['REVIEW']:.1f}%", delta_color="off")
st.caption("Percentages are of all inspected units. REVIEW = classifier confidence below the threshold "
           "(needs a human check). V1 rows are frame samples, not unique products.")

hourly = per_hour(df)
left, mid, right = st.columns(3)
with left:
    st.subheader("OK / NG share")
    pie = pd.DataFrame({"result": list(s["n"].keys()), "count": list(s["n"].values())})
    st.plotly_chart(px.pie(pie, names="result", values="count", color="result",
                           color_discrete_map={"OK": "#3cc83c", "NG": "#eb3232", "REVIEW": "#ffaa00"}),
                    use_container_width=True)
with mid:
    st.subheader("Units per hour")
    st.plotly_chart(px.bar(hourly, x="hour", y=["OK", "NG", "REVIEW"], barmode="stack",
                           labels={"value": "Units", "hour": "Hour", "variable": "Result"},
                           color_discrete_map={"OK": "#3cc83c", "NG": "#eb3232", "REVIEW": "#ffaa00"}),
                    use_container_width=True)
with right:
    st.subheader("Hourly NG rate")
    st.plotly_chart(px.line(hourly, x="hour", y="ng_rate_pct", markers=True,
                            labels={"ng_rate_pct": "NG rate (%)", "hour": "Hour"}), use_container_width=True)

st.subheader("Inspection history")
cols = [c for c in ["id", "timestamp", "result", "prediction", "confidence", "det_score", "votes", "source", "image_path"]
        if c in df.columns]
show = df[cols].copy()
show["timestamp"] = show["timestamp"].astype(str)
st.dataframe(show, use_container_width=True, hide_index=True)
st.download_button("Download inspection log (CSV)", data=df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="window_ac_dent_inspections.csv", mime="text/csv")

st.subheader("Image evidence")
selected = st.selectbox("Choose inspection ID", df["id"].astype(int).tolist())
row = df[df["id"] == selected].iloc[0]
img_path = Path(str(row["image_path"])) if pd.notna(row["image_path"]) else None
if img_path and img_path.exists():
    st.image(str(img_path), caption=f"ID {selected} | {row['result']} | confidence {float(row['confidence']):.1%}",
             use_container_width=True)
else:
    st.caption("Image evidence path is missing or file is not available.")
