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

INDICATOR_CONFIG = {
    "peak_x_norm": {
        "label": "Posicion relativa del pico",
        "range": (-0.5, 0.5),
    },
    "area_symmetry": {
        "label": "Simetria interna de area",
        "range": (-1.0, 1.0),
    },
    "centroid_x_norm": {
        "label": "Posicion relativa del centroide",
        "range": (-0.5, 0.5),
    },
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
    .video-metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 0.75rem 0 1.5rem;
    }
    .video-metric-card {
        min-height: 168px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(164, 185, 194, 0.28);
        border-radius: 7px;
        background: rgb(24, 36, 42);
        display: flex;
        flex-direction: column;
    }
    .video-metric-label {
        color: rgba(250, 250, 250, 0.94);
        font-size: 0.92rem;
        font-weight: 700;
    }
    .video-metric-value {
        margin-top: 0.7rem;
        color: rgb(250, 250, 250);
        font-size: 2rem;
        line-height: 1.15;
    }
    .video-metric-value.is-text {
        font-size: 1.45rem;
    }
    .video-metric-help {
        margin-top: auto;
        padding-top: 1rem;
        color: rgb(187, 219, 236);
        font-size: 0.82rem;
        line-height: 1.35;
    }
    @media (max-width: 1050px) {
        .video-metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 620px) {
        .video-metric-grid {
            grid-template-columns: minmax(0, 1fr);
        }
        .video-metric-card {
            min-height: 148px;
        }
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


def optional_numeric_value(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


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


def valid_indicator_values(features_df: pd.DataFrame, indicator: str) -> pd.Series:
    if features_df.empty or indicator not in features_df.columns:
        return pd.Series(dtype=float)
    valid = features_df
    if "is_valid" in valid.columns:
        valid_flags = (
            valid["is_valid"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "si", "yes"])
        )
        valid = valid[valid_flags]
    return pd.to_numeric(valid[indicator], errors="coerce").dropna()


def render_indicator_summary(video: pd.Series | None, features_df: pd.DataFrame) -> None:
    rows = []
    for indicator, config in INDICATOR_CONFIG.items():
        values = valid_indicator_values(features_df, indicator)
        if video is not None:
            p25 = numeric_value(video.get(f"p25_{indicator}"), float("nan"))
            median = numeric_value(video.get(f"median_{indicator}"), float("nan"))
            p75 = numeric_value(video.get(f"p75_{indicator}"), float("nan"))
        elif values.empty:
            p25 = median = p75 = float("nan")
        else:
            p25, median, p75 = values.quantile([0.25, 0.50, 0.75]).tolist()
        rows.append(
            {
                "Indicador": config["label"],
                "Mediana": median,
                "Rango central (P25 - P75)": f"{p25:.4f} a {p75:.4f}",
            }
        )
    table = pd.DataFrame(rows)
    table["Mediana"] = pd.to_numeric(table["Mediana"], errors="coerce").round(4)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "Negativo indica tendencia visual interna hacia la izquierda; positivo, "
        "hacia la derecha. El rango central contiene la mitad intermedia de los frames validos."
    )


def render_indicator_distributions(features_df: pd.DataFrame) -> None:
    with st.expander("Ver distribuciones de los indicadores"):
        columns = st.columns(3)
        for column, (indicator, config) in zip(columns, INDICATOR_CONFIG.items()):
            with column:
                st.markdown(f"**{config['label']}**")
                values = valid_indicator_values(features_df, indicator)
                if values.empty:
                    st.write("Sin datos validos.")
                    continue
                minimum, maximum = config["range"]
                clipped = values.clip(lower=minimum, upper=maximum)
                observed_min = float(clipped.min())
                observed_max = float(clipped.max())
                observed_span = observed_max - observed_min
                padding = max(observed_span * 0.15, (maximum - minimum) * 0.02)
                chart_min = max(minimum, observed_min - padding)
                chart_max = min(maximum, observed_max + padding)
                if chart_max <= chart_min:
                    chart_min = max(minimum, observed_min - padding)
                    chart_max = min(maximum, observed_max + padding)
                bin_count = min(8, max(3, int(len(values) ** 0.5) + 1))
                bin_width = (chart_max - chart_min) / bin_count
                bin_edges = [
                    chart_min + index * bin_width for index in range(bin_count + 1)
                ]
                intervals = pd.cut(
                    clipped,
                    bins=bin_edges,
                    include_lowest=True,
                )
                counts = intervals.value_counts(sort=False)
                labels = [
                    f"{bin_edges[index]:.3f} a {bin_edges[index + 1]:.3f}"
                    for index in range(bin_count)
                ]
                chart = pd.DataFrame(
                    {
                        "intervalo": labels,
                        "frames": counts.values,
                    }
                ).set_index("intervalo")
                st.bar_chart(chart, height=220)
        st.caption(
            "La escala visual se ajusta al rango observado y considera solo frames "
            "validos. Los CSV mantienen histogramas fijos de 20 intervalos para "
            "analisis y comparacion."
        )


def render_frame_details(features_df: pd.DataFrame) -> None:
    table = display_frame_table(features_df)
    frame_count = len(table)
    with st.expander(f"Ver detalle por frame ({frame_count})"):
        if table.empty:
            st.write("Sin resultados por frame.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)


def metric_card(
    label: str,
    value: str,
    description: str,
    *,
    text_value: bool = False,
) -> str:
    value_class = "video-metric-value is-text" if text_value else "video-metric-value"
    return (
        '<div class="video-metric-card">'
        f'<div class="video-metric-label">{html.escape(label)}</div>'
        f'<div class="{value_class}">{html.escape(value)}</div>'
        f'<div class="video-metric-help">{html.escape(description)}</div>'
        "</div>"
    )


def render_verdict(video: pd.Series | None, summary: dict) -> None:
    st.subheader("Indicadores generales del video")
    if video is None:
        st.write("Sin veredicto disponible.")
        return

    analytic_rules = summary.get("analytic_rules", {})
    if not isinstance(analytic_rules, dict):
        analytic_rules = {}
    threshold_peak = numeric_value(analytic_rules.get("threshold_peak"), 0.06)
    threshold_area = numeric_value(analytic_rules.get("threshold_area"), 0.07)
    final_verdict = text_value(video.get("final_verdict"), "sin_veredicto")
    label = VERDICT_LABELS.get(final_verdict, final_verdict)
    confidence = numeric_value(video.get("confidence")) * 100.0
    valid_ratio = numeric_value(video.get("valid_ratio"))
    cards = [
        metric_card(
            "Frames extraidos",
            text_value(video.get("analytic_rows"), "0"),
            "Imagenes tomadas del video segun el intervalo configurado.",
        ),
        metric_card(
            "Frames validos",
            text_value(video.get("valid_frames"), "0"),
            "Frames con mascara completa y perfil suficiente para evaluar.",
        ),
        metric_card(
            "No evaluables",
            text_value(video.get("no_evaluable_count"), "0"),
            "Frames descartados por falta de deteccion o contacto con bordes.",
        ),
        metric_card(
            "Confianza",
            f"{confidence:.1f}%",
            "Escala 0% a 100%: proporcion de la clase dominante entre los frames validos.",
        ),
        metric_card(
            "Centrado",
            f"{numeric_value(video.get('centrado_ratio')):.1%}",
            "Rango 0% a 100%: proporcion de frames validos clasificados como centrados.",
        ),
        metric_card(
            "Izquierda",
            f"{numeric_value(video.get('izquierda_ratio')):.1%}",
            "Rango 0% a 100%: proporcion de frames validos con tendencia hacia la izquierda.",
        ),
        metric_card(
            "Derecha",
            f"{numeric_value(video.get('derecha_ratio')):.1%}",
            "Rango 0% a 100%: proporcion de frames validos con tendencia hacia la derecha.",
        ),
        metric_card(
            "Veredicto",
            label.upper(),
            "Posibles: centrada, izquierda, derecha, mixta o no evaluable.",
            text_value=True,
        ),
        metric_card(
            "Pico relativo mediano",
            f"{numeric_value(video.get('median_peak_x_norm')):.3f}",
            (
                f"Rango -0.5 a 0.5: < -{threshold_peak:.2f} izquierda; "
                f"-{threshold_peak:.2f} a {threshold_peak:.2f} centro; "
                f"> {threshold_peak:.2f} derecha."
            ),
        ),
        metric_card(
            "Simetria interna mediana",
            f"{numeric_value(video.get('median_area_symmetry')):.3f}",
            (
                f"Rango -1 a 1: < -{threshold_area:.2f} izquierda; "
                f"-{threshold_area:.2f} a {threshold_area:.2f} simetrica; "
                f"> {threshold_area:.2f} derecha."
            ),
        ),
        metric_card(
            "Centroide relativo mediano",
            f"{numeric_value(video.get('median_centroid_x_norm')):.3f}",
            (
                "Rango -0.5 a 0.5: negativo es izquierda, 0 es centro y positivo "
                "es derecha. Es descriptivo."
            ),
        ),
        metric_card(
            "Frames evaluables",
            f"{valid_ratio:.1%}",
            "Rango 0% a 100%: porcentaje de frames con un resultado valido.",
        ),
    ]
    st.markdown(
        '<div class="video-metric-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def render_inference_time(summary: dict) -> None:
    st.subheader("Tiempo de inferencia")
    timings = summary.get("timings", {})
    if not isinstance(timings, dict):
        timings = {}
    mean_ms = optional_numeric_value(
        timings.get("yolo_inference_mean_ms_per_image")
    )
    total_ms = optional_numeric_value(timings.get("yolo_inference_total_ms"))
    wall_seconds = optional_numeric_value(
        timings.get("yolo_prediction_wall_seconds")
    )
    if mean_ms is None:
        st.caption(
            "No disponible para esta corrida anterior. Se registrara al ejecutar "
            "nuevamente el pipeline."
        )
        return

    columns = st.columns(3)
    columns[0].metric("Inferencia YOLO por frame", f"{mean_ms:.1f} ms")
    columns[1].metric(
        "Inferencia YOLO acumulada",
        f"{(total_ms or 0.0):.1f} ms",
    )
    columns[2].metric(
        "Generacion de mascaras",
        f"{(wall_seconds or 0.0):.2f} s",
    )
    st.caption(
        "La inferencia es reportada por Ultralytics. La generacion de mascaras "
        "incluye prediccion, postproceso y escritura para todos los frames de la corrida."
    )


def render_statistics(video: pd.Series | None, features_df: pd.DataFrame) -> None:
    st.subheader("Distribucion de los indicadores")
    render_indicator_summary(video, features_df)
    render_indicator_distributions(features_df)
    render_frame_details(features_df)


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

render_verdict(video, summary)
render_inference_time(summary)
render_statistics(video, active_features)

st.subheader("Descargas")
download_cols = st.columns(3)
with download_cols[0]:
    file_download("Resumen por video", paths.video_summary_csv, "text/csv")
with download_cols[1]:
    file_download("Resultados por frame", paths.features_csv, "text/csv")
with download_cols[2]:
    file_download("Resumen de indicadores", paths.indicator_summary_csv, "text/csv")

download_cols = st.columns(3)
with download_cols[0]:
    file_download(
        "Distribuciones de indicadores",
        paths.indicator_distributions_csv,
        "text/csv",
    )
with download_cols[1]:
    file_download(
        "Resumen tecnico de la corrida",
        paths.pipeline_summary_json,
        "application/json",
    )
with download_cols[2]:
    if paths.result_dir.exists():
        st.caption(f"Resultados: {relative_to_project(paths.result_dir)}")

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
