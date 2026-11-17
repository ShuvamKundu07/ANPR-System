"""
app.py — ANPR POC Streamlit Dashboard (Phase 8 + Phase 9)
──────────────────────────────────────────────────────────
Entry point.  Run with:
    streamlit run app.py

Navigation via sidebar:
    • Live Monitoring      – real-time video / webcam ANPR
    • Vehicle Records      – browse / search stored observations
    • Hotlist              – manage flagged plates
    • Alerts               – view and acknowledge hotlist-match alerts
    • Movement History     – cross-camera plate timeline (search by plate)
    • Two-Camera Demo      – Phase 9 two-camera simulation
"""

import streamlit as st
from anpr.detector import ensure_model_weights

st.set_page_config(
    page_title="ANPR — License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure YOLO detection model weights exist (auto-downloads if deploying to a new cloud container)
ensure_model_weights()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 ANPR POC")
    st.caption("Automatic License Plate Recognition")
    st.divider()
    page = st.radio(
        "Navigation",
        [
            "📹 Live Monitoring",
            "📋 Vehicle Records",
            "🚨 Hotlist",
            "🔔 Alerts",
            "🗺️ Movement History",
            "🎬 Two-Camera Demo",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Academic POC — not for production use.")

# ── Route to view ──────────────────────────────────────────────────────────────
from dashboard.views import (
    alerts_view,
    hotlist_view,
    live_monitoring_view,
    movement_history_view,
    records_view,
    two_camera_demo_view,
)

if page == "📹 Live Monitoring":
    live_monitoring_view()
elif page == "📋 Vehicle Records":
    records_view()
elif page == "🚨 Hotlist":
    hotlist_view()
elif page == "🔔 Alerts":
    alerts_view()
elif page == "🗺️ Movement History":
    movement_history_view()
elif page == "🎬 Two-Camera Demo":
    two_camera_demo_view()
