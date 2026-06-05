from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

WEB_DIR = Path(__file__).resolve().parent
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

from src.pipeline_runner import PipelineParams, relative_to_project, run_pipeline
from src.run_storage import (
    count_files,
    default_run_name,
    list_runs,
    overlay_paths,
    paths_for_run,
    prepare_uploads,
)


st.set_page_config(
    page_title="Cono de Carga - Pipeline",
    layout="wide",
)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def file_download(label: str, path: Path, mime: str) -> None:
    if not path.exists():
        return
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        use_container_width=True,
    )


def status_item(label: str, done: bool, detail: str = "") -> None:
    icon = "[OK]" if done else "[--]"
    st.write(f"{icon} {label}" + (f" - {detail}" if detail else ""))


st.title("Cono de Carga")

with st.sidebar:
    st.subheader("Nueva corrida")
    run_name = st.text_input("run_name", value=default_run_name())
    uploaded_files = st.file_uploader(
        "Videos",
        type=["mp4", "avi", "mov", "mkv", "m4v"],
        accept_multiple_files=True,
    )
    interval_seconds = st.number_input(
        "interval_seconds",
        min_value=0.0,
        value=2.0,
        step=0.5,
    )
    max_frames_raw = st.number_input(
        "max_frames_per_video (0 = sin limite)",
        min_value=0,
        value=20,
        step=5,
    )
    conf = st.slider("YOLO conf", min_value=0.05, max_value=0.95, value=0.35, step=0.05)
    device = st.text_input("device", value="0")
    start_docker = st.checkbox("levantar Docker", value=False)
    video_min_valid_ratio = st.slider(
        "min valid ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    video_majority_ratio = st.slider(
        "majority ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
    )
    run_clicked = st.button("Ejecutar pipeline", type="primary", use_container_width=True)

    st.divider()
    existing_runs = list_runs()
    selected_run = st.selectbox("Corrida", options=existing_runs) if existing_runs else None


if run_clicked:
    if not uploaded_files:
        st.error("Sube al menos un video.")
        st.stop()

    paths = paths_for_run(run_name)
    saved_count = prepare_uploads(paths, uploaded_files)
    params = PipelineParams(
        interval_seconds=float(interval_seconds),
        max_frames_per_video=None if int(max_frames_raw) == 0 else int(max_frames_raw),
        conf=float(conf),
        device=device,
        start_docker=bool(start_docker),
        video_min_valid_ratio=float(video_min_valid_ratio),
        video_majority_ratio=float(video_majority_ratio),
    )
    with st.spinner("Ejecutando pipeline"):
        return_code, log = run_pipeline(paths, params)
    st.code(log, language="text")
    if return_code != 0:
        st.error(f"Pipeline fallo con codigo {return_code}.")
        st.stop()
    st.success(f"Pipeline terminado. Videos cargados: {saved_count}.")
    selected_run = paths.run_name


if not selected_run:
    st.info("No hay corridas disponibles.")
    st.stop()

paths = paths_for_run(selected_run)
summary = read_json(paths.pipeline_summary_json)
video_df = read_csv(paths.video_summary_csv)
features_df = read_csv(paths.features_csv)
overlays = overlay_paths(paths)

st.subheader(selected_run)

cols = st.columns(5)
with cols[0]:
    status_item("videos", paths.videos_dir.exists(), f"{count_files(paths.videos_dir, '*')}")
with cols[1]:
    status_item("frames", (paths.work_dir / "frames").exists(), f"{count_files(paths.work_dir / 'frames', '*.jpg')}")
with cols[2]:
    status_item("mascaras", (paths.work_dir / "yolo_masks").exists(), f"{count_files(paths.work_dir / 'yolo_masks', '*.png')}")
with cols[3]:
    status_item("analitico", paths.features_csv.exists())
with cols[4]:
    status_item("veredicto", paths.video_summary_csv.exists())

if summary:
    metric_cols = st.columns(4)
    metric_cols[0].metric("videos", summary.get("video_count", 0))
    metric_cols[1].metric("frames", summary.get("frame_count", 0))
    feature_summary = summary.get("feature_summary", {})
    metric_cols[2].metric("validos", feature_summary.get("valid", 0))
    metric_cols[3].metric("invalidos", feature_summary.get("invalid", 0))

if not video_df.empty:
    st.subheader("Veredicto por video")
    st.dataframe(video_df, use_container_width=True, hide_index=True)

if not features_df.empty:
    st.subheader("Resultados por frame")
    st.dataframe(features_df, use_container_width=True, hide_index=True)

st.subheader("Overlays")
if overlays:
    max_images = st.slider("cantidad", min_value=1, max_value=len(overlays), value=min(12, len(overlays)))
    grid_cols = st.columns(3)
    for idx, overlay in enumerate(overlays[:max_images]):
        with grid_cols[idx % 3]:
            st.image(str(overlay), caption=overlay.name, use_container_width=True)
else:
    st.write("Sin overlays.")

st.subheader("Descargas")
download_cols = st.columns(4)
with download_cols[0]:
    file_download("video_summary.csv", paths.video_summary_csv, "text/csv")
with download_cols[1]:
    file_download("features.csv", paths.features_csv, "text/csv")
with download_cols[2]:
    file_download("pipeline_summary.json", paths.pipeline_summary_json, "application/json")
with download_cols[3]:
    if paths.result_dir.exists():
        st.write(relative_to_project(paths.result_dir))
