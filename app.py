# 🔴 FIX PyTorch + Streamlit conflict (VERY TOP)
import torch
torch.classes.__path__ = []
import streamlit as st
import time
import cv2
import pandas as pd
from may import FeedManager   # change this to your actual file

# ✅ MUST be first Streamlit command
st.set_page_config(layout="wide")

# 🔁 Video sources
sources = {
    "feed_1": "videoplayback.mp4",
    "feed_2": "video2.mp4",
    "feed_3": "video3.mp4"
}

# ✅ Start system ONCE
@st.cache_resource
def start_system():
    manager = FeedManager(sources)
    manager.start()
    return manager

manager = start_system()

# Color mapping for different object classes
CLASS_COLORS = {
    "Drone": "background-color: #ff4d4d",
    "Bird": "background-color: #fff176"
}

def color_threat_row(row):
    """Apply color styling based on threat class."""
    color = CLASS_COLORS.get(row["Class"], "background-color: #81c784")
    return [color] * len(row)

# 🔥 Placeholder container for live updates
placeholder = st.empty()

# 🔁 LIVE LOOP (NO rerun, NO freeze)
while True:

    with manager.lock:
        f1 = manager.latest_frames.get("feed_1")
        f2 = manager.latest_frames.get("feed_2")
        f3 = manager.latest_frames.get("feed_3")

    with placeholder.container():

        col1, col2, col3, col4 = st.columns([1, 1, 1, 0.7])

        # 🟢 Feed 1
        if f1 is not None:
            col1.image(cv2.resize(f1, (400, 300)), channels="BGR")
        else:
            col1.write("Waiting for feed 1...")

        # 🟢 Feed 2
        if f2 is not None:
            col2.image(cv2.resize(f2, (400, 300)), channels="BGR")
        else:
            col2.write("Waiting for feed 2...")

        # 🟢 Feed 3
        if f3 is not None:
            col3.image(cv2.resize(f3, (400, 300)), channels="BGR")
        else:
            col3.write("Waiting for feed 3...")

        # 🔴 Threat Table
        threats = []

        for name, processor in manager.processors.items():
            for entry in list(processor.log_data)[-10:]:
                threats.append({
                    "Feed": name,
                    "ID": entry["id"],
                    "Class": entry["class"],
                    "Confidence": round(entry["confidence"], 2),
                    "Time": round(entry.get("video_time_sec", 0), 2)
                })

        col4.subheader("🚨 Threat Table")
        df = pd.DataFrame(threats)

        if df.empty:
            col4.write("No detections yet...")
        else:
            styled_df = df.style.apply(color_threat_row, axis=1)
            col4.dataframe(styled_df, use_container_width=True)

    # 🔥 Refresh rate (balance performance)
    time.sleep(0.15)