from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

WEB_DIR = Path(__file__).resolve().parent
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

from src.pipeline_runner import PipelineParams, relative_to_project, run_pipeline_stream
from src.run_storage import (
    count_files,
    default_run_name,
    list_runs,
    overlay_paths,
    paths_for_run,
    prepare_uploads,
)


FRAME_TABLE_COLUMNS = [
    "image_name",
    "timestamp_sec",
    "predicted_class",
    "is_valid",
    "error_reason",
    "peak_x_norm",
    "area_symmetry",
    "profile_roughness",
    "num_peaks",
    "centroid_x_norm",
    "mask_area_px",
]

NUMERIC_COLUMNS = [
    "timestamp_sec",
    "peak_x_norm",
    "area_symmetry",
    "profile_roughness",
    "num_peaks",
    "centroid_x_norm",
    "mask_area_px",
]

VERDICT_LABELS = {
    "carga_centrada": "Carga centrada",
    "tendencia_izquierda": "Tendencia izquierda",
    "tendencia_derecha": "Tendencia derecha",
    "mixto_o_inestable": "Mixto o inestable",
    "no_evaluable": "No evaluable",
}

st.set_page_config(
    page_title="Cono de Carga - Pipeline",
    layout="wide",
)

st.markdown(
    """
    <style>
    .console-box {
        max-height: 220px;
        overflow: auto;
        border: 1px solid rgba(250, 250, 250, 0.12);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        background: rgba(17, 24, 39, 0.9);
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.82rem;
        line-height: 1.45;
        white-space: pre;
    }
    .muted-small {
        color: rgba(250, 250, 250, 0.65);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
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


def text_value(value: object, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value)
    return text if text else default


def numeric_value(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def last_log_line(log: str) -> str:
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def render_console(log: str, target: object | None = None) -> None:
    root = target or st
    escaped_log = html.escape(log)
    escaped_last = html.escape(last_log_line(log))
    with root.container():
        st.markdown("Consola")
        if escaped_last:
            st.caption(f"Ultima salida: {escaped_last}")
        st.markdown(
            f'<div class="console-box">{escaped_log}</div>',
            unsafe_allow_html=True,
        )


def video_label(row: pd.Series) -> str:
    name = text_value(row.get("video_name"))
    video_id = text_value(row.get("video_id"))
    return name or video_id or "Video"


def selected_video_row(video_df: pd.DataFrame, video_id: str | None) -> pd.Series | None:
    if video_df.empty:
        return None
    if video_id and "video_id" in video_df.columns:
        rows = video_df[video_df["video_id"].astype(str) == video_id]
        if not rows.empty:
            return rows.iloc[0]
    return video_df.iloc[0]


def frame_names_for_video(frame_df: pd.DataFrame, video_id: str | None) -> set[str]:
    if frame_df.empty or "frame_name" not in frame_df.columns:
        return set()
    if video_id and "video_id" in frame_df.columns:
        frame_df = frame_df[frame_df["video_id"].astype(str) == video_id]
    return set(frame_df["frame_name"].astype(str))


def filter_features(features_df: pd.DataFrame, frame_names: set[str]) -> pd.DataFrame:
    if features_df.empty or "image_name" not in features_df.columns or not frame_names:
        return features_df.copy()
    return features_df[features_df["image_name"].astype(str).isin(frame_names)].copy()


def enrich_features(features_df: pd.DataFrame, frame_df: pd.DataFrame) -> pd.DataFrame:
    if features_df.empty or frame_df.empty:
        return features_df.copy()
    if "image_name" not in features_df.columns or "frame_name" not in frame_df.columns:
        return features_df.copy()
    manifest_cols = [
        col
        for col in ["frame_name", "timestamp_sec", "frame_index", "video_id", "video_name"]
        if col in frame_df.columns
    ]
    return features_df.merge(
        frame_df[manifest_cols],
        left_on="image_name",
        right_on="frame_name",
        how="left",
    )


def filter_overlays(overlays: list[Path], frame_names: set[str]) -> list[Path]:
    if not frame_names:
        return overlays
    expected = {f"{Path(name).stem}_overlay.jpg" for name in frame_names}
    return [path for path in overlays if path.name in expected]


def display_frame_table(features_df: pd.DataFrame) -> pd.DataFrame:
    if features_df.empty:
        return features_df
    cols = [col for col in FRAME_TABLE_COLUMNS if col in features_df.columns]
    table = features_df[cols].copy()
    for col in NUMERIC_COLUMNS:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").round(4)
    return table


def render_verdict(video: pd.Series | None) -> None:
    st.subheader("Veredicto")
    if video is None:
        st.write("Sin veredicto disponible.")
        return

    final_verdict = text_value(video.get("final_verdict"), "sin_veredicto")
    label = VERDICT_LABELS.get(final_verdict, final_verdict)
    confidence = numeric_value(video.get("confidence"))
    valid_ratio = numeric_value(video.get("valid_ratio"))

    cols = st.columns(4)
    cols[0].metric("veredicto", label)
    cols[1].metric("confianza", f"{confidence:.2f}")
    cols[2].metric("frames validos", text_value(video.get("valid_frames"), "0"))
    cols[3].metric("ratio valido", f"{valid_ratio:.2f}")


def render_statistics(video: pd.Series | None, features_df: pd.DataFrame) -> None:
    st.subheader("Estadisticas")
    if video is not None:
        cols = st.columns(5)
        cols[0].metric("frames", text_value(video.get("analytic_rows"), "0"))
        cols[1].metric("centrado", text_value(video.get("centrado_count"), "0"))
        cols[2].metric("izquierda", text_value(video.get("izquierda_count"), "0"))
        cols[3].metric("derecha", text_value(video.get("derecha_count"), "0"))
        cols[4].metric("no evaluable", text_value(video.get("no_evaluable_count"), "0"))

        geo_cols = st.columns(4)
        geo_cols[0].metric("peak mediano", f"{numeric_value(video.get('median_peak_x_norm')):.3f}")
        geo_cols[1].metric("simetria mediana", f"{numeric_value(video.get('median_area_symmetry')):.3f}")
        geo_cols[2].metric("rugosidad media", f"{numeric_value(video.get('mean_profile_roughness')):.3f}")
        geo_cols[3].metric("picos medios", f"{numeric_value(video.get('mean_num_peaks')):.2f}")

    table = display_frame_table(features_df)
    if table.empty:
        st.write("Sin resultados por frame.")
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)


if "last_log" not in st.session_state:
    st.session_state["last_log"] = ""
if "last_log_run" not in st.session_state:
    st.session_state["last_log_run"] = ""


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
    console_slot = st.empty()
    status_slot = st.empty()
    status_slot.info("Ejecutando pipeline...")

    def update_console(log_text: str) -> None:
        st.session_state["last_log"] = log_text
        st.session_state["last_log_run"] = paths.run_name
        render_console(log_text, console_slot)

    return_code, log = run_pipeline_stream(paths, params, update_console)

    st.session_state["last_log"] = log
    st.session_state["last_log_run"] = paths.run_name
    selected_run = paths.run_name

    if return_code != 0:
        status_slot.empty()
        st.error(f"Pipeline fallo con codigo {return_code}.")
        st.stop()

    status_message = f"Pipeline terminado. Videos cargados: {saved_count}."
    status_slot.success(status_message)


if not selected_run:
    if st.session_state["last_log"]:
        render_console(st.session_state["last_log"])
    st.info("No hay corridas disponibles.")
    st.stop()

paths = paths_for_run(selected_run)
summary = read_json(paths.pipeline_summary_json)
video_df = read_csv(paths.video_summary_csv)
frame_df = read_csv(paths.frame_manifest_csv)
features_df = enrich_features(read_csv(paths.features_csv), frame_df)
all_overlays = overlay_paths(paths)

selected_video_id = None
if not video_df.empty and "video_id" in video_df.columns:
    video_options = video_df["video_id"].astype(str).tolist()
    selected_video_id = video_options[0]
    if len(video_options) > 1:
        label_by_id = {
            str(row["video_id"]): video_label(row)
            for _, row in video_df.iterrows()
        }
        with st.sidebar:
            selected_video_id = st.selectbox(
                "Video",
                options=video_options,
                format_func=lambda value: label_by_id.get(value, value),
            )

video = selected_video_row(video_df, selected_video_id)
video_title = video_label(video) if video is not None else selected_run
frame_names = frame_names_for_video(frame_df, selected_video_id)
active_features = filter_features(features_df, frame_names)
active_overlays = filter_overlays(all_overlays, frame_names)

st.header(video_title)
st.caption(f"Corrida: {selected_run}")

st.subheader("Overlays")
if active_overlays:
    max_images = st.slider(
        "cantidad",
        min_value=1,
        max_value=len(active_overlays),
        value=min(12, len(active_overlays)),
    )
    grid_cols = st.columns(3)
    for idx, overlay in enumerate(active_overlays[:max_images]):
        with grid_cols[idx % 3]:
            st.image(str(overlay), caption=overlay.name, use_container_width=True)
else:
    st.write("Sin overlays.")

render_verdict(video)
render_statistics(video, active_features)

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

with st.expander("Estado tecnico de la corrida"):
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
        st.json(summary, expanded=False)
