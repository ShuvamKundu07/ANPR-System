"""
dashboard/views.py
──────────────────
Phase 8 + Phase 9 — Streamlit Dashboard View Functions.

Six sections, each implemented as a standalone function:
  • live_monitoring_view()      – Process video/webcam and show annotated frames.
  • records_view()              – Browse / search vehicle observation records.
  • hotlist_view()              – Add, list, and remove hotlist entries.
  • alerts_view()               – Display and acknowledge hotlist-match alerts.
  • movement_history_view()     – Show chronological cross-camera plate history.
  • two_camera_demo_view()      – Phase 9: simulate two cameras and show journey.

Design rules:
  - All DB and pipeline logic stays in anpr/; views only call those APIs.
  - No absolute paths or machine-specific constants.
  - Every section shows a helpful message when no data exists yet.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import streamlit as st

from anpr.config import (
    CAM_ID_1,
    CAM_ID_2,
    CROP_DIR,
    DB_PATH,
    DUPLICATE_WINDOW_SEC,
    PROJECT_ROOT,
)
from anpr.database import Database
from anpr.pipeline import ANPRPipeline
from anpr.plate_utils import normalize_plate


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_db() -> Database:
    """Return (or create) a cached Database instance stored in session state."""
    if "db" not in st.session_state:
        st.session_state["db"] = Database()
    return st.session_state["db"]


def _fmt_conf(value: float | None) -> str:
    """Format a confidence float as a percentage string."""
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def _abs_image_path(rel_path: str | None) -> Path | None:
    """Resolve a stored (possibly relative) image path to an absolute Path."""
    if not rel_path:
        return None
    return Database.to_absolute_image_path(rel_path)


def _list_input_videos() -> list[str]:
    """Return video filenames found under data/input_videos/."""
    video_dir = PROJECT_ROOT / "data" / "input_videos"
    if not video_dir.exists():
        return []
    return sorted(
        p.name
        for p in video_dir.glob("*.*")
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )


# ── 1. Live Monitoring ─────────────────────────────────────────────────────────

def live_monitoring_view() -> None:
    """Live video/webcam monitoring with plate detection and OCR."""
    st.header("📹 Live Monitoring")

    col_cfg, col_disp = st.columns([1, 2])

    with col_cfg:
        st.subheader("Configuration")
        source_type = st.radio("Source", ["Video File", "Webcam"], horizontal=True)

        cam_or_path: str | int | None = None

        if source_type == "Video File":
            uploaded = st.file_uploader(
                "Upload a video", type=["mp4", "avi", "mov", "mkv"]
            )
            existing = _list_input_videos()
            default_ix = 0
            options = ["— select —"] + existing
            if "demo_vehicle.mp4" in existing:
                default_ix = options.index("demo_vehicle.mp4")

            chosen_existing = st.selectbox(
                "Or choose sample video",
                options=options,
                index=default_ix,
            )
            if uploaded:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded.name).suffix
                )
                tmp.write(uploaded.read())
                tmp.flush()
                cam_or_path = tmp.name
            elif chosen_existing != "— select —":
                cam_or_path = str(
                    PROJECT_ROOT / "data" / "input_videos" / chosen_existing
                )
        else:
            cam_index = st.number_input("Webcam index", min_value=0, max_value=10, value=0)
            cam_or_path = int(cam_index)
            st.info(
                "ℹ️ **Note on Webcams:** Accesses a hardware camera connected directly to the server host. "
                "In cloud deployments (like Streamlit Cloud), the cloud VM does not have a physical webcam. "
                "Please choose **Video File** to select the bundled sample or upload an MP4."
            )

        camera_id = st.text_input("Camera ID", value="CAM001").strip() or "CAM001"
        det_conf = st.slider("Detection confidence", 0.1, 1.0, 0.4, 0.05)
        ocr_conf = st.slider("OCR confidence threshold", 0.05, 1.0, 0.15, 0.05)
        dup_win = st.number_input(
            "Duplicate-suppression window (sec)", min_value=1, max_value=300, value=30
        )
        max_frames = st.number_input(
            "Max frames (0 = all)", min_value=0, max_value=10000, value=300
        )

    with col_disp:
        st.subheader("Live Feed")

        if "monitoring_running" not in st.session_state:
            st.session_state["monitoring_running"] = False

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_clicked = st.button(
                "▶ Start", disabled=st.session_state["monitoring_running"]
            )
        with btn_col2:
            stop_clicked = st.button(
                "■ Stop", disabled=not st.session_state["monitoring_running"]
            )

        if stop_clicked:
            st.session_state["monitoring_running"] = False
            st.info("Monitoring stopped.")

        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()

        if start_clicked:
            if cam_or_path is None:
                st.error("Select a video file or webcam source first.")
            else:
                st.session_state["monitoring_running"] = True
                db = _get_db()
                pipeline = ANPRPipeline(
                    camera_id=camera_id,
                    db=db,
                    detector_conf=det_conf,
                    ocr_conf_thresh=ocr_conf,
                    duplicate_window_sec=int(dup_win),
                )

                total_frames = 0
                plates_seen: list[str] = []

                try:
                    for result in pipeline.process_video(
                        source=cam_or_path,
                        max_frames=int(max_frames) if max_frames else 0,
                    ):
                        if not st.session_state.get("monitoring_running", False):
                            break

                        total_frames += 1
                        rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb, use_container_width=True)

                        for obs in result.observations:
                            if obs.is_valid_plate and obs.plate_number:
                                plates_seen.append(obs.plate_number)

                        obs_summary = []
                        for obs in result.observations:
                            tag = obs.plate_number or "?"
                            if obs.is_hotlist_match:
                                tag = f"🔴 {tag} (HOTLIST)"
                            elif obs.is_valid_plate:
                                tag = f"🟢 {tag} ({_fmt_conf(obs.ocr_result.confidence)})"
                            else:
                                tag = "🟡 Low OCR"
                            obs_summary.append(tag)

                        status_placeholder.markdown(
                            f"**Frame {total_frames}** | "
                            + (", ".join(obs_summary) if obs_summary else "No plate detected")
                        )

                        with metrics_placeholder.container():
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Frames processed", total_frames)
                            m2.metric("Plates logged", len(plates_seen))
                            m3.metric("Last plate", plates_seen[-1] if plates_seen else "—")

                except Exception as exc:
                    err_msg = str(exc)
                    if isinstance(cam_or_path, int) and "Could not open video source" in err_msg:
                        st.error(
                            f"❌ Could not open webcam index `{cam_or_path}`. "
                            "When hosted on the cloud (e.g. Streamlit Cloud), the cloud server "
                            "does not have an attached physical webcam device. Please switch Source to "
                            "**Video File** to use the bundled sample clip or upload your own video."
                        )
                    else:
                        st.error(f"Pipeline error: {exc}")
                finally:
                    st.session_state["monitoring_running"] = False
                    status_placeholder.info("Processing complete.")


# ── 2. Vehicle Records ─────────────────────────────────────────────────────────

def records_view() -> None:
    """Browse, filter, and search vehicle observations."""
    st.header("📋 Vehicle Records")
    db = _get_db()

    with st.expander("🔍 Search & Filter", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            search_plate = st.text_input("Plate number contains")
        with fcol2:
            filter_cam = st.text_input("Camera ID")
        with fcol3:
            limit = st.number_input("Max results", min_value=1, max_value=500, value=50)

    records = db.search_observations(
        plate_number=search_plate.strip() if search_plate else None,
        camera_id=filter_cam.strip() if filter_cam else None,
        limit=int(limit),
    )

    if not records:
        st.info("No records found. Run Live Monitoring to capture some observations.")
        return

    df = pd.DataFrame(records)
    display_cols = ["id", "plate_number", "plate_text_raw", "camera_id",
                    "detected_at", "ocr_confidence", "detection_confidence"]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].copy()
    df_display["ocr_confidence"] = df_display["ocr_confidence"].apply(
        lambda v: f"{float(v)*100:.1f}%"
    )
    df_display["detection_confidence"] = df_display["detection_confidence"].apply(
        lambda v: f"{float(v)*100:.1f}%"
    )
    df_display.rename(columns={
        "id": "ID", "plate_number": "Plate (Normalized)", "plate_text_raw": "Raw OCR",
        "camera_id": "Camera", "detected_at": "Timestamp",
        "ocr_confidence": "OCR Conf", "detection_confidence": "Det Conf",
    }, inplace=True)

    st.dataframe(df_display, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(records)} record(s).")

    st.subheader("View Plate Crop")
    record_ids = [r["id"] for r in records]
    sel_id = st.selectbox("Select record ID to view crop", options=["—"] + record_ids)
    if sel_id != "—":
        row = next((r for r in records if r["id"] == int(sel_id)), None)
        if row:
            img_path = _abs_image_path(row.get("image_path"))
            if img_path and img_path.exists():
                st.image(str(img_path), caption=f"Plate crop for record {sel_id}", width=320)
            else:
                st.warning("Crop image not found on disk.")
            st.json({
                "plate_number": row.get("plate_number"),
                "raw_ocr": row.get("plate_text_raw"),
                "camera": row.get("camera_id"),
                "timestamp": row.get("detected_at"),
                "ocr_conf": _fmt_conf(row.get("ocr_confidence")),
                "det_conf": _fmt_conf(row.get("detection_confidence")),
            })


# ── 3. Hotlist Management ──────────────────────────────────────────────────────

def hotlist_view() -> None:
    """Add, list, and remove hotlist entries."""
    st.header("🚨 Hotlist Management")
    db = _get_db()

    with st.expander("➕ Add / Update Entry", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            new_plate = st.text_input("Plate number")
            new_status = st.selectbox(
                "Status", ["STOLEN", "WANTED", "SUSPECT", "EXPIRED", "OTHER"]
            )
        with col2:
            new_desc = st.text_area("Description (optional)", height=80)

        if st.button("Add to Hotlist"):
            raw = new_plate.strip()
            if not raw:
                st.warning("Enter a plate number first.")
            else:
                normalized = normalize_plate(raw)
                if not normalized:
                    st.error("Could not normalize the plate number.")
                else:
                    try:
                        db.add_hotlist_entry(
                            plate_number=normalized,
                            status=new_status,
                            description=new_desc.strip() or None,
                        )
                        st.success(f"✅ Added **{normalized}** ({new_status}) to hotlist.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to add entry: {exc}")

    st.subheader("Current Hotlist Entries")
    entries = db.list_hotlist_entries()
    if not entries:
        st.info("Hotlist is empty. Add entries above.")
        return

    df = pd.DataFrame(entries)[["id", "plate_number", "status", "description", "created_at"]]
    df.rename(columns={
        "id": "ID", "plate_number": "Plate", "status": "Status",
        "description": "Description", "created_at": "Added At",
    }, inplace=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"{len(entries)} entry/entries in hotlist.")

    st.subheader("Remove Entry")
    plates = [e["plate_number"] for e in entries]
    to_remove = st.selectbox("Select plate to remove", ["— select —"] + plates)
    if st.button("Remove", type="secondary"):
        if to_remove == "— select —":
            st.warning("Choose a plate first.")
        else:
            removed = db.remove_hotlist_entry(to_remove)
            if removed:
                st.success(f"Removed **{to_remove}** from hotlist.")
                st.rerun()
            else:
                st.error("Entry not found.")


# ── 4. Alerts ─────────────────────────────────────────────────────────────────

def alerts_view() -> None:
    """Display hotlist-match alerts and allow acknowledgement."""
    st.header("🔔 Alerts")
    db = _get_db()

    filter_ack = st.radio(
        "Show", ["All", "Unacknowledged only", "Acknowledged only"], horizontal=True
    )
    ack_filter: bool | None = None
    if filter_ack == "Unacknowledged only":
        ack_filter = False
    elif filter_ack == "Acknowledged only":
        ack_filter = True

    alerts = db.list_alerts(acknowledged=ack_filter, limit=100)

    if not alerts:
        st.info("No alerts found.")
        return

    unacked = sum(1 for a in alerts if not a.get("acknowledged"))
    if unacked:
        st.error(f"⚠️  {unacked} unacknowledged alert(s) require attention.")

    for alert in alerts:
        is_acked = bool(alert.get("acknowledged"))
        status_icon = "✅" if is_acked else "🔴"

        with st.container(border=True):
            hdr_col, ack_col = st.columns([4, 1])
            with hdr_col:
                st.markdown(
                    f"{status_icon} **{alert['plate_number']}** — "
                    f"{alert.get('alert_type', '?')} | "
                    f"Camera: `{alert.get('camera_id', '?')}` | "
                    f"{alert.get('created_at', '')}"
                )
                st.caption(alert.get("message", ""))
                ocr_conf = alert.get("ocr_confidence")
                if ocr_conf is not None:
                    st.caption(f"OCR Confidence: {_fmt_conf(ocr_conf)}")
            with ack_col:
                if not is_acked:
                    if st.button("Acknowledge", key=f"ack_{alert['id']}"):
                        db.acknowledge_alert(alert["id"])
                        st.success("Acknowledged.")
                        st.rerun()

            img_path = _abs_image_path(alert.get("image_path"))
            if img_path and img_path.exists():
                st.image(str(img_path), caption="Plate crop", width=240)
            else:
                st.caption("_(No crop image available)_")


# ── 5. Movement History ────────────────────────────────────────────────────────

def movement_history_view() -> None:
    """Search a plate and display its cross-camera observation history."""
    st.header("🗺️ Movement History")
    st.caption(
        "Shows all observations of a plate ordered by timestamp. "
        "Plate identity is based on OCR — confirm independently for legal purposes."
    )

    db = _get_db()

    search_col, btn_col = st.columns([3, 1])
    with search_col:
        plate_query = st.text_input("Enter plate number to search")
    with btn_col:
        st.write("")
        search_btn = st.button("🔍 Search", use_container_width=True)

    if not (search_btn and plate_query.strip()):
        st.info("Enter a plate number and press Search.")
        return

    history = db.get_movement_history(plate_query.strip())
    if not history:
        st.warning(
            f"No observations found for plate: **{normalize_plate(plate_query.strip())}**"
        )
        return

    normalized = normalize_plate(plate_query.strip())
    st.success(f"Found **{len(history)}** observation(s) for plate **{normalized}**")

    st.subheader("Observation Timeline")
    for i, row in enumerate(history, start=1):
        with st.container(border=True):
            tcol, icol = st.columns([3, 1])
            with tcol:
                st.markdown(
                    f"**#{i}** &nbsp; `{row['camera_id']}` — {row['detected_at']}"
                )
                st.caption(
                    f"OCR: `{row['plate_text_raw']}` → `{row['plate_number']}` | "
                    f"OCR conf: {_fmt_conf(row['ocr_confidence'])} | "
                    f"Det conf: {_fmt_conf(row['detection_confidence'])}"
                )
            with icol:
                img_path = _abs_image_path(row.get("image_path"))
                if img_path and img_path.exists():
                    st.image(str(img_path), width=120)

    st.subheader("Summary Table")
    df = pd.DataFrame(history)[
        ["camera_id", "detected_at", "plate_number", "ocr_confidence", "image_path"]
    ].copy()
    df["ocr_confidence"] = df["ocr_confidence"].apply(lambda v: f"{float(v)*100:.1f}%")
    df.drop(columns=["image_path"], inplace=True)
    df.rename(columns={
        "camera_id": "Camera", "detected_at": "Timestamp",
        "plate_number": "Plate (Normalized)", "ocr_confidence": "OCR Conf",
    }, inplace=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    cam_summary = df.groupby("Camera")["Timestamp"].agg(
        First="min", Last="max", Count="count"
    ).reset_index()
    if len(cam_summary) > 1:
        st.subheader("Cross-Camera Summary")
        st.dataframe(cam_summary, use_container_width=True, hide_index=True)
        st.caption(
            "⚠️ This is an inferred observation history based on plate recognition. "
            "Observations from different cameras are linked by OCR text only."
        )


# ── 6. Two-Camera Demo (Phase 9) ───────────────────────────────────────────────

# Human-readable location labels for each simulated camera
_CAM_LOCATIONS: dict[str, str] = {
    CAM_ID_1: "Entry Gate — Sector 12",
    CAM_ID_2: "Exit Gate — Sector 14",
}


def two_camera_demo_view() -> None:
    """
    Phase 9 — Two-Camera Movement History Demonstration.

    Runs the ANPR pipeline twice over the same video, assigning different
    camera IDs and staggered timestamps, then displays the cross-camera
    observation history for every plate that appeared on both cameras.

    This is a controlled simulation, not live visual multi-camera tracking.
    """
    st.header("🎬 Two-Camera Movement Demo")
    st.info(
        "**How it works:** The same video is processed twice — once as **CAM001** "
        f"({_CAM_LOCATIONS[CAM_ID_1]}) and again as **CAM002** "
        f"({_CAM_LOCATIONS[CAM_ID_2]}) with timestamps offset by 12 minutes. "
        "This simulates a vehicle travelling between two camera checkpoints. "
        "Observations are linked by normalized OCR plate text only."
    )

    # ── Video selection ────────────────────────────────────────────────────────
    st.subheader("1. Select Video")
    existing = _list_input_videos()
    if not existing:
        st.warning(
            "No videos found in `data/input_videos/`. "
            "Upload a video via Live Monitoring or place one in that folder."
        )
        uploaded = st.file_uploader("Upload a video for the demo", type=["mp4", "avi", "mov", "mkv"])
        if not uploaded:
            return
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        video_path = tmp.name
    else:
        chosen = st.selectbox("Choose video", existing)
        video_path = str(PROJECT_ROOT / "data" / "input_videos" / chosen)

    # ── Simulation settings ────────────────────────────────────────────────────
    st.subheader("2. Simulation Settings")
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        cam1_id = st.text_input("Camera 1 ID", value=CAM_ID_1)
        cam1_loc = st.text_input("Camera 1 Location", value=_CAM_LOCATIONS[CAM_ID_1])
    with scol2:
        cam2_id = st.text_input("Camera 2 ID", value=CAM_ID_2)
        cam2_loc = st.text_input("Camera 2 Location", value=_CAM_LOCATIONS[CAM_ID_2])
    with scol3:
        offset_min = st.number_input(
            "Time offset between cameras (minutes)", min_value=1, max_value=120, value=12
        )
        max_frames = st.number_input(
            "Max frames per camera", min_value=10, max_value=2000, value=150
        )
        det_conf = st.slider("Detection confidence", 0.1, 1.0, 0.4, 0.05)

    cam1_id = cam1_id.strip() or CAM_ID_1
    cam2_id = cam2_id.strip() or CAM_ID_2
    cam_locations = {cam1_id: cam1_loc, cam2_id: cam2_loc}

    # ── Run simulation ─────────────────────────────────────────────────────────
    st.subheader("3. Run Simulation")
    run_btn = st.button("▶ Run Two-Camera Simulation", type="primary")

    if not run_btn:
        st.caption("Press the button to process the video through both simulated cameras.")
        # Still show any existing cross-camera data from the DB
        _show_cross_camera_history(_get_db(), cam_locations)
        return

    from datetime import datetime, timedelta, timezone

    cam1_start = datetime(2026, 11, 17, 10, 30, 0, tzinfo=timezone.utc)
    cam2_start = cam1_start + timedelta(minutes=int(offset_min))

    db = _get_db()

    # Load models once — reused for both cameras
    progress = st.progress(0, text="Loading models…")

    try:
        from anpr.detector import LicensePlateDetector
        from anpr.ocr import PlateOCR
        from anpr.video import frame_generator, open_source

        detector = LicensePlateDetector(confidence_threshold=det_conf)
        ocr = PlateOCR(gpu=False)

        progress.progress(10, text="Models loaded. Processing CAM1…")

        # ── CAM1 pass ──────────────────────────────────────────────────────────
        pipeline1 = ANPRPipeline(
            camera_id=cam1_id, db=db, detector=detector, ocr=ocr,
            duplicate_window_sec=5,
        )
        cam1_plates = _run_simulation_pass(
            pipeline=pipeline1,
            video_path=video_path,
            start_time=cam1_start,
            max_frames=int(max_frames),
            progress_base=10,
            progress_end=50,
            progress_obj=progress,
            label=f"{cam1_id} ({cam1_loc})",
        )

        progress.progress(50, text=f"CAM1 done ({len(cam1_plates)} plates). Processing CAM2…")

        # ── CAM2 pass ──────────────────────────────────────────────────────────
        pipeline2 = ANPRPipeline(
            camera_id=cam2_id, db=db, detector=detector, ocr=ocr,
            duplicate_window_sec=5,
        )
        cam2_plates = _run_simulation_pass(
            pipeline=pipeline2,
            video_path=video_path,
            start_time=cam2_start,
            max_frames=int(max_frames),
            progress_base=50,
            progress_end=95,
            progress_obj=progress,
            label=f"{cam2_id} ({cam2_loc})",
        )

        progress.progress(100, text="Simulation complete.")

    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        return

    # ── Results summary ────────────────────────────────────────────────────────
    st.subheader("4. Results")
    shared = set(cam1_plates) & set(cam2_plates)
    rcol1, rcol2, rcol3 = st.columns(3)
    rcol1.metric(f"{cam1_id} plates", len(set(cam1_plates)))
    rcol2.metric(f"{cam2_id} plates", len(set(cam2_plates)))
    rcol3.metric("Cross-camera matches", len(shared))

    if shared:
        st.success(
            f"✅ **{len(shared)} plate(s) detected on both cameras:** "
            + ", ".join(f"`{p}`" for p in sorted(shared))
        )
    else:
        st.warning(
            "No plates were detected on both cameras in this run. "
            "Try a video that contains clearly visible license plates, "
            "or lower the detection confidence threshold."
        )

    _show_cross_camera_history(db, cam_locations)


def _run_simulation_pass(
    pipeline: ANPRPipeline,
    video_path: str,
    start_time: "datetime",
    max_frames: int,
    progress_base: int,
    progress_end: int,
    progress_obj: "st.delta_generator.DeltaGenerator",
    label: str,
) -> list[str]:
    """
    Process a video through one pipeline pass with synthetic timestamps.

    Returns list of unique normalized plate numbers recorded.
    """
    from datetime import timedelta
    from anpr.video import frame_generator, open_source

    plates: list[str] = []
    fps = 25.0

    with open_source(video_path) as cap:
        # Try to read actual FPS from video
        actual_fps = cap.get(2)  # cv2.CAP_PROP_FPS == 5
        import cv2
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps and actual_fps > 0:
            fps = actual_fps

        total = max_frames
        for idx, frame in frame_generator(cap, max_frames=max_frames):
            ts = (start_time + timedelta(seconds=idx / fps)).strftime("%Y-%m-%d %H:%M:%S")
            result = pipeline.process_frame(frame, frame_index=idx, timestamp=ts)

            for obs in result.observations:
                if obs.is_valid_plate and not obs.is_duplicate and obs.plate_number:
                    plates.append(obs.plate_number)

            # Update progress bar
            pct = progress_base + int((idx / max(total, 1)) * (progress_end - progress_base))
            progress_obj.progress(
                min(pct, progress_end),
                text=f"Processing {label} — frame {idx + 1}/{max_frames}…",
            )

    return plates


def _show_cross_camera_history(db: Database, cam_locations: dict[str, str]) -> None:
    """Display the cross-camera movement history from the database."""
    all_records = db.get_recent_observations(limit=1000)
    if not all_records:
        return

    st.subheader("5. Cross-Camera Movement History")
    st.caption(
        "⚠️ **Limitation:** Plate identity is inferred from OCR text. "
        "Different physical vehicles with OCR errors may be incorrectly linked. "
        "This is not a visual vehicle re-identification system."
    )

    # Group by plate
    by_plate: dict[str, list[dict]] = defaultdict(list)
    for rec in all_records:
        by_plate[rec["plate_number"]].append(rec)

    for plate in by_plate:
        by_plate[plate].sort(key=lambda r: r["detected_at"])

    multi_cam = {
        p: recs
        for p, recs in by_plate.items()
        if len({r["camera_id"] for r in recs}) > 1
    }

    if not multi_cam:
        st.info(
            "No plates found on more than one camera yet. "
            "Run the simulation above, or use Live Monitoring with different Camera IDs."
        )
        return

    st.success(f"Found **{len(multi_cam)}** plate(s) with cross-camera history.")

    for plate, recs in sorted(multi_cam.items()):
        cams = sorted({r["camera_id"] for r in recs})
        with st.expander(
            f"🚗 `{plate}` — cameras: {', '.join(cams)} ({len(recs)} observations)",
            expanded=True,
        ):
            for i, rec in enumerate(recs, 1):
                cam = rec["camera_id"]
                loc = cam_locations.get(cam, cam)
                img_path = _abs_image_path(rec.get("image_path"))

                rcol, icol = st.columns([4, 1])
                with rcol:
                    st.markdown(
                        f"**Step {i}** &nbsp;|&nbsp; "
                        f"`{cam}` — **{loc}** &nbsp;|&nbsp; "
                        f"🕐 {rec['detected_at']}"
                    )
                    st.caption(
                        f"Raw OCR: `{rec['plate_text_raw']}` → Normalized: `{rec['plate_number']}` | "
                        f"OCR conf: {_fmt_conf(rec['ocr_confidence'])}"
                    )
                with icol:
                    if img_path and img_path.exists():
                        st.image(str(img_path), width=110)

            # Summary table for this plate
            df = pd.DataFrame(recs)[["camera_id", "detected_at", "ocr_confidence"]].copy()
            df["camera_id"] = df["camera_id"].apply(
                lambda c: f"{c} ({cam_locations.get(c, c)})"
            )
            df["ocr_confidence"] = df["ocr_confidence"].apply(
                lambda v: f"{float(v)*100:.1f}%"
            )
            df.rename(columns={
                "camera_id": "Camera / Location",
                "detected_at": "Timestamp",
                "ocr_confidence": "OCR Conf",
            }, inplace=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
