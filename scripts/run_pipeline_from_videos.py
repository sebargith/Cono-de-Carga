from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cone_load.statistics import (  # noqa: E402
    indicator_distribution_rows,
    indicator_summary_rows,
    summarize_indicators,
)

DEFAULT_MODEL = (
    "perception/models/reference/"
    "carga_visible_yolov8n-seg_20260527_195509/best.pt"
)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
CLASS_CENTERED = "centrado"
CLASS_LEFT = "izquierda"
CLASS_RIGHT = "derecha"
CLASS_NO_EVALUABLE = "no_evaluable"
VERDICT_CENTERED = "carga_centrada"
VERDICT_LEFT = "tendencia_izquierda"
VERDICT_RIGHT = "tendencia_derecha"
VERDICT_MIXED = "mixto_o_inestable"
VERDICT_NO_EVALUABLE = "no_evaluable"
VIDEO_SUMMARY_COLUMNS = [
    "video_name",
    "video_id",
    "video_path",
    "total_frames",
    "analytic_rows",
    "valid_frames",
    "invalid_frames",
    "valid_ratio",
    "centrado_count",
    "izquierda_count",
    "derecha_count",
    "no_evaluable_count",
    "centrado_ratio",
    "izquierda_ratio",
    "derecha_ratio",
    "p25_peak_x_norm",
    "median_peak_x_norm",
    "p75_peak_x_norm",
    "mean_peak_x_norm",
    "p25_area_symmetry",
    "median_area_symmetry",
    "p75_area_symmetry",
    "mean_area_symmetry",
    "p25_centroid_x_norm",
    "median_centroid_x_norm",
    "p75_centroid_x_norm",
    "mean_centroid_x_norm",
    "mean_profile_roughness",
    "mean_num_peaks",
    "final_verdict",
    "confidence",
    "verdict_reason",
]
INDICATOR_SUMMARY_COLUMNS = [
    "scope_type",
    "scope_id",
    "scope_name",
    "indicator",
    "indicator_label",
    "count",
    "p25",
    "median",
    "p75",
]
INDICATOR_DISTRIBUTION_COLUMNS = [
    "scope_type",
    "scope_id",
    "scope_name",
    "indicator",
    "indicator_label",
    "bin_index",
    "bin_start",
    "bin_end",
    "count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline desde videos locales: extrae frames, genera mascaras YOLO "
            "compatibles con el analitico y corre el analitico geometrico."
        )
    )
    parser.add_argument("--videos-dir", required=True, help="Carpeta con videos de entrada.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre del experimento. Por defecto usa timestamp.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Carpeta de trabajo visible para Docker. Por defecto queda dentro "
            "de perception/model_testing/pipeline_runs/. La web app puede usar "
            "web_app/outputs/runs/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Carpeta final de resultados. Por defecto queda dentro de "
            "outputs/pipeline_runs/. La web app puede usar web_app/outputs/runs/."
        ),
    )
    parser.add_argument(
        "--perception-dir",
        default="perception",
        help="Directorio del modulo de percepcion con docker-compose.yml.",
    )
    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_MODEL,
        help="Ruta al modelo YOLO-seg. Debe quedar dentro del directorio perception.",
    )
    parser.add_argument(
        "--analytic-config",
        default="configs/analytic_config.yaml",
        help="Configuracion del analitico geometrico.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="Extrae 1 frame cada N segundos. Usa 0 para extraer todos los frames.",
    )
    parser.add_argument("--jpg-quality", type=int, default=90)
    parser.add_argument("--recursive", action="store_true", help="Buscar videos recursivamente.")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--mask-mode",
        choices=("largest", "merge_all"),
        default="largest",
        help="Como combinar instancias YOLO antes de pasar al analitico.",
    )
    parser.add_argument(
        "--docker-service",
        default="cone-yolo",
        help="Servicio Docker Compose usado para correr YOLO.",
    )
    parser.add_argument(
        "--start-docker",
        action="store_true",
        help="Ejecuta docker compose up -d antes de correr YOLO.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Borra la corrida previa con el mismo run-name si existe.",
    )
    parser.add_argument(
        "--video-min-valid-ratio",
        type=float,
        default=0.5,
        help="Minima fraccion de frames validos para emitir veredicto por video.",
    )
    parser.add_argument(
        "--video-majority-ratio",
        type=float,
        default=0.6,
        help="Fraccion minima de frames validos que deben apoyar una clase dominante.",
    )
    return parser.parse_args()


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def require_dir(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"No existe {name}: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} no es un directorio: {path}")
    return path


def require_file(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"No existe {name}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} no es un archivo: {path}")
    return path


def prepare_clean_dir(
    path: Path,
    allowed_bases: list[Path],
    overwrite: bool,
    label: str,
) -> Path:
    path = path.resolve()
    resolved_bases = [base.resolve() for base in allowed_bases]
    is_allowed = any(is_relative_to(path, base) and path != base for base in resolved_bases)
    if not is_allowed:
        bases = ", ".join(str(base) for base in resolved_bases)
        raise ValueError(f"{label} debe quedar dentro de una de estas rutas: {bases}. Ruta: {path}")

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{label} existe pero no es un directorio: {path}")

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{label} ya existe y no esta vacio: {path}. Usa --overwrite "
                "o elige otro --run-name."
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return path


def default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def list_videos(videos_dir: Path, recursive: bool) -> list[Path]:
    iterator = videos_dir.rglob("*") if recursive else videos_dir.glob("*")
    allowed = {ext.lower() for ext in VIDEO_EXTENSIONS}
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in allowed)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "video"


def video_id(video_path: Path, videos_dir: Path) -> str:
    try:
        rel = video_path.relative_to(videos_dir)
        stem = "__".join(rel.with_suffix("").parts)
    except ValueError:
        stem = video_path.stem
    return safe_name(stem)


def frame_step_from_interval(fps: float, interval_seconds: float) -> int:
    if interval_seconds <= 0:
        return 1
    if fps <= 0:
        return 1
    return max(int(round(fps * interval_seconds)), 1)


def write_manifest_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def nested_value(config: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def extract_frames(
    videos: list[Path],
    videos_dir: Path,
    frames_dir: Path,
    interval_seconds: float,
    jpg_quality: int,
    max_frames_per_video: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
    used_frame_names: set[str] = set()

    for idx, video_path in enumerate(videos, start=1):
        cap = cv2.VideoCapture(str(video_path))
        current_video_id = video_id(video_path, videos_dir)
        if not cap.isOpened():
            video_rows.append(
                {
                    "video_name": video_path.name,
                    "video_path": str(video_path),
                    "video_id": current_video_id,
                    "fps": "",
                    "total_frames": "",
                    "saved_frames": 0,
                    "status": "error_no_se_pudo_abrir_video",
                }
            )
            print(f"[{idx}/{len(videos)}] No se pudo abrir: {video_path}")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = frame_step_from_interval(fps, interval_seconds)
        frame_index = 0
        saved = 0

        print(f"[{idx}/{len(videos)}] Extrayendo frames: {video_path.name}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            should_save = interval_seconds <= 0 or frame_index % step == 0
            if should_save:
                frame_name = f"{current_video_id}__frame{frame_index:06d}.jpg"
                if frame_name in used_frame_names:
                    raise ValueError(f"Nombre de frame duplicado: {frame_name}")
                used_frame_names.add(frame_name)

                out_path = frames_dir / frame_name
                if not cv2.imwrite(str(out_path), frame, encode_param):
                    cap.release()
                    raise OSError(f"No se pudo escribir frame: {out_path}")

                timestamp_sec = frame_index / fps if fps > 0 else ""
                frame_rows.append(
                    {
                        "video_name": video_path.name,
                        "video_path": str(video_path),
                        "video_id": current_video_id,
                        "frame_name": frame_name,
                        "frame_path": str(out_path),
                        "frame_index": frame_index,
                        "timestamp_sec": (
                            "" if timestamp_sec == "" else f"{timestamp_sec:.6f}"
                        ),
                    }
                )
                saved += 1
                if max_frames_per_video is not None and saved >= max_frames_per_video:
                    break

            frame_index += 1

        cap.release()
        video_rows.append(
            {
                "video_name": video_path.name,
                "video_path": str(video_path),
                "video_id": current_video_id,
                "fps": f"{fps:.6f}" if fps > 0 else "",
                "total_frames": total_frames,
                "saved_frames": saved,
                "status": "ok" if saved > 0 else "sin_frames_extraidos",
            }
        )
        print(f"  Frames guardados: {saved}")

    return frame_rows, video_rows


def perception_host_path(value: str | Path, perception_dir: Path) -> Path:
    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([ROOT / raw, perception_dir / raw])

    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            if is_relative_to(resolved, perception_dir):
                return resolved

    candidate_list = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No se pudo resolver la ruta dentro de perception. "
        f"Valor: {value}. Candidatos: {candidate_list}"
    )


def to_container_path(host_path: Path, perception_dir: Path, web_app_dir: Path | None = None) -> str:
    resolved = host_path.resolve()
    perception_root = perception_dir.resolve()
    mount_points: list[tuple[Path, str]] = [(perception_root, "/workspace")]
    if web_app_dir is not None:
        mount_points.append((web_app_dir.resolve(), "/workspace/web_app"))

    for host_root, container_root in mount_points:
        if is_relative_to(resolved, host_root):
            rel = resolved.relative_to(host_root)
            suffix = rel.as_posix()
            return container_root if not suffix else f"{container_root}/{suffix}"

    raise ValueError(f"La ruta no esta montada en Docker: {resolved}")


def run_command(command: list[str], cwd: Path, label: str) -> None:
    print(f"\n== {label} ==")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"Fallo {label} con codigo {completed.returncode}")


def run_yolo_masks(
    perception_dir: Path,
    web_app_dir: Path | None,
    docker_service: str,
    yolo_model: Path,
    frames_dir: Path,
    masks_dir: Path,
    conf: float,
    imgsz: int | None,
    device: str,
    mask_mode: str,
    start_docker: bool,
) -> None:
    if start_docker:
        run_command(["docker", "compose", "up", "-d"], perception_dir, "levantar Docker")

    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        docker_service,
        "python",
        "scripts/predict_to_analytic_masks.py",
        "--model",
        to_container_path(yolo_model, perception_dir, web_app_dir),
        "--images-dir",
        to_container_path(frames_dir, perception_dir, web_app_dir),
        "--masks-dir",
        to_container_path(masks_dir, perception_dir, web_app_dir),
        "--conf",
        str(conf),
        "--device",
        str(device),
        "--mask-mode",
        mask_mode,
        "--overwrite",
    ]
    if imgsz is not None:
        command.extend(["--imgsz", str(imgsz)])

    run_command(command, perception_dir, "YOLO a mascaras analiticas")


def run_analytic(frames_dir: Path, masks_dir: Path, analytic_dir: Path, config_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_simple_analytic.py"),
        "--images-dir",
        str(frames_dir),
        "--masks-dir",
        str(masks_dir),
        "--output-dir",
        str(analytic_dir),
        "--config",
        str(config_path),
    ]
    run_command(command, ROOT, "analitico geometrico")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bool_from_csv(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "si"}


def finite_float_from_csv(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def median_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2
    if n % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def ratio(count: int, denom: int) -> float:
    return float(count / denom) if denom > 0 else 0.0


def decide_video_verdict(
    valid_ratio: float,
    centered_ratio: float,
    left_ratio: float,
    right_ratio: float,
    median_peak_x_norm: float,
    median_area_symmetry: float,
    min_valid_ratio: float,
    majority_ratio: float,
    threshold_peak: float,
    threshold_area: float,
) -> tuple[str, float, str]:
    class_confidence = max(centered_ratio, left_ratio, right_ratio)
    if valid_ratio < min_valid_ratio:
        return VERDICT_NO_EVALUABLE, valid_ratio, "valid_ratio_bajo"

    if left_ratio >= majority_ratio:
        return VERDICT_LEFT, left_ratio, "mayoria_frames_izquierda"
    if right_ratio >= majority_ratio:
        return VERDICT_RIGHT, right_ratio, "mayoria_frames_derecha"
    if centered_ratio >= majority_ratio:
        return VERDICT_CENTERED, centered_ratio, "mayoria_frames_centrado"

    peak_ok = math.isfinite(median_peak_x_norm)
    area_ok = math.isfinite(median_area_symmetry)
    if peak_ok and area_ok:
        if (
            abs(median_peak_x_norm) < threshold_peak
            and abs(median_area_symmetry) < threshold_area
        ):
            return VERDICT_CENTERED, class_confidence, "medianas_geometricas_centradas"
        if median_peak_x_norm < -threshold_peak or median_area_symmetry < -threshold_area:
            return VERDICT_LEFT, class_confidence, "medianas_geometricas_izquierda"
        if median_peak_x_norm > threshold_peak or median_area_symmetry > threshold_area:
            return VERDICT_RIGHT, class_confidence, "medianas_geometricas_derecha"

    return VERDICT_MIXED, class_confidence, "sin_mayoria_clara"


def build_video_summary(
    frame_manifest_rows: list[dict[str, Any]],
    video_manifest_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, str]],
    analytic_config: dict[str, Any],
    min_valid_ratio: float,
    majority_ratio: float,
) -> list[dict[str, Any]]:
    frame_to_video = {
        str(row["frame_name"]): {
            "video_name": str(row["video_name"]),
            "video_id": str(row["video_id"]),
            "video_path": str(row["video_path"]),
        }
        for row in frame_manifest_rows
    }
    by_video: dict[str, dict[str, Any]] = {}
    for video_row in video_manifest_rows:
        current_video_id = str(video_row["video_id"])
        by_video[current_video_id] = {
            "video_name": str(video_row["video_name"]),
            "video_id": current_video_id,
            "video_path": str(video_row["video_path"]),
            "total_frames": int(video_row.get("saved_frames") or 0),
            "features": [],
        }

    for feature in feature_rows:
        frame_name = str(feature.get("image_name", ""))
        frame_info = frame_to_video.get(frame_name)
        if frame_info is None:
            continue
        current_video_id = frame_info["video_id"]
        if current_video_id not in by_video:
            by_video[current_video_id] = {
                "video_name": frame_info["video_name"],
                "video_id": current_video_id,
                "video_path": frame_info["video_path"],
                "total_frames": 0,
                "features": [],
            }
        by_video[current_video_id]["features"].append(feature)

    threshold_peak = float(nested_value(analytic_config, ("rules", "threshold_peak"), 0.12))
    threshold_area = float(nested_value(analytic_config, ("rules", "threshold_area"), 0.12))
    rows: list[dict[str, Any]] = []
    for video_id_key in sorted(by_video):
        item = by_video[video_id_key]
        features = list(item["features"])
        total_frames = int(item["total_frames"])
        analytic_rows = len(features)
        valid_features = [row for row in features if bool_from_csv(row.get("is_valid", ""))]
        valid_frames = len(valid_features)
        invalid_frames = max(total_frames - valid_frames, 0)
        valid_ratio = ratio(valid_frames, total_frames)
        class_counts = Counter(row.get("predicted_class", "") for row in features)
        centered_count = int(class_counts.get(CLASS_CENTERED, 0))
        left_count = int(class_counts.get(CLASS_LEFT, 0))
        right_count = int(class_counts.get(CLASS_RIGHT, 0))
        no_evaluable_count = int(class_counts.get(CLASS_NO_EVALUABLE, 0))
        centered_ratio = ratio(centered_count, valid_frames)
        left_ratio = ratio(left_count, valid_frames)
        right_ratio = ratio(right_count, valid_frames)

        peak_values = [
            value
            for value in (finite_float_from_csv(row.get("peak_x_norm")) for row in valid_features)
            if value is not None
        ]
        area_values = [
            value
            for value in (finite_float_from_csv(row.get("area_symmetry")) for row in valid_features)
            if value is not None
        ]
        centroid_values = [
            value
            for value in (
                finite_float_from_csv(row.get("centroid_x_norm")) for row in valid_features
            )
            if value is not None
        ]
        roughness_values = [
            value
            for value in (
                finite_float_from_csv(row.get("profile_roughness")) for row in valid_features
            )
            if value is not None
        ]
        num_peak_values = [
            value
            for value in (finite_float_from_csv(row.get("num_peaks")) for row in valid_features)
            if value is not None
        ]
        median_peak = median_or_nan(peak_values)
        mean_peak = mean_or_nan(peak_values)
        median_area = median_or_nan(area_values)
        mean_area = mean_or_nan(area_values)
        mean_centroid = mean_or_nan(centroid_values)
        mean_roughness = mean_or_nan(roughness_values)
        mean_num_peaks = mean_or_nan(num_peak_values)
        indicator_stats = summarize_indicators(valid_features, valid_only=False)
        peak_stats = indicator_stats["peak_x_norm"]
        area_stats = indicator_stats["area_symmetry"]
        centroid_stats = indicator_stats["centroid_x_norm"]
        final_verdict, confidence, verdict_reason = decide_video_verdict(
            valid_ratio=valid_ratio,
            centered_ratio=centered_ratio,
            left_ratio=left_ratio,
            right_ratio=right_ratio,
            median_peak_x_norm=median_peak,
            median_area_symmetry=median_area,
            min_valid_ratio=min_valid_ratio,
            majority_ratio=majority_ratio,
            threshold_peak=threshold_peak,
            threshold_area=threshold_area,
        )

        rows.append(
            {
                "video_name": item["video_name"],
                "video_id": item["video_id"],
                "video_path": item["video_path"],
                "total_frames": total_frames,
                "analytic_rows": analytic_rows,
                "valid_frames": valid_frames,
                "invalid_frames": invalid_frames,
                "valid_ratio": fmt_float(valid_ratio),
                "centrado_count": centered_count,
                "izquierda_count": left_count,
                "derecha_count": right_count,
                "no_evaluable_count": no_evaluable_count,
                "centrado_ratio": fmt_float(centered_ratio),
                "izquierda_ratio": fmt_float(left_ratio),
                "derecha_ratio": fmt_float(right_ratio),
                "p25_peak_x_norm": fmt_float(float(peak_stats["p25"])),
                "median_peak_x_norm": fmt_float(median_peak),
                "p75_peak_x_norm": fmt_float(float(peak_stats["p75"])),
                "mean_peak_x_norm": fmt_float(mean_peak),
                "p25_area_symmetry": fmt_float(float(area_stats["p25"])),
                "median_area_symmetry": fmt_float(median_area),
                "p75_area_symmetry": fmt_float(float(area_stats["p75"])),
                "mean_area_symmetry": fmt_float(mean_area),
                "p25_centroid_x_norm": fmt_float(float(centroid_stats["p25"])),
                "median_centroid_x_norm": fmt_float(float(centroid_stats["median"])),
                "p75_centroid_x_norm": fmt_float(float(centroid_stats["p75"])),
                "mean_centroid_x_norm": fmt_float(mean_centroid),
                "mean_profile_roughness": fmt_float(mean_roughness),
                "mean_num_peaks": fmt_float(mean_num_peaks),
                "final_verdict": final_verdict,
                "confidence": fmt_float(confidence),
                "verdict_reason": verdict_reason,
            }
        )

    return rows


def build_indicator_outputs(
    frame_manifest_rows: list[dict[str, Any]],
    video_manifest_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, str]],
    run_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = indicator_summary_rows(
        feature_rows,
        scope_type="run",
        scope_id=run_name,
        scope_name=run_name,
    )
    distribution_rows = indicator_distribution_rows(
        feature_rows,
        scope_type="run",
        scope_id=run_name,
        scope_name=run_name,
    )

    frame_to_video = {
        str(row["frame_name"]): str(row["video_id"]) for row in frame_manifest_rows
    }
    features_by_video: dict[str, list[dict[str, str]]] = {}
    for feature in feature_rows:
        video_id = frame_to_video.get(str(feature.get("image_name", "")))
        if video_id is not None:
            features_by_video.setdefault(video_id, []).append(feature)

    for video in video_manifest_rows:
        video_id = str(video["video_id"])
        video_name = str(video["video_name"])
        rows = features_by_video.get(video_id, [])
        summary_rows.extend(
            indicator_summary_rows(
                rows,
                scope_type="video",
                scope_id=video_id,
                scope_name=video_name,
            )
        )
        distribution_rows.extend(
            indicator_distribution_rows(
                rows,
                scope_type="video",
                scope_id=video_id,
                scope_name=video_name,
            )
        )

    for row in summary_rows:
        for field in ("p25", "median", "p75"):
            row[field] = fmt_float(float(row[field]))
    for row in distribution_rows:
        row["bin_start"] = fmt_float(float(row["bin_start"]))
        row["bin_end"] = fmt_float(float(row["bin_end"]))
    return summary_rows, distribution_rows


def json_safe_indicator_summary(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    summaries = summarize_indicators(rows)
    output: dict[str, dict[str, Any]] = {}
    for indicator, values in summaries.items():
        output[indicator] = {}
        for key, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                output[indicator][key] = None
            else:
                output[indicator][key] = value
    return output


def summarize_features(features_csv: Path) -> dict[str, Any]:
    rows = read_csv_rows(features_csv)
    class_counts = Counter(row.get("predicted_class", "") for row in rows)
    valid_rows = [row for row in rows if bool_from_csv(row.get("is_valid", ""))]
    invalid_reasons = Counter(
        row.get("error_reason", "")
        for row in rows
        if not bool_from_csv(row.get("is_valid", ""))
    )
    return {
        "rows": len(rows),
        "valid": len(valid_rows),
        "invalid": len(rows) - len(valid_rows),
        "predicted_class_counts": dict(class_counts),
        "invalid_reason_counts": dict(invalid_reasons),
        "indicator_statistics": json_safe_indicator_summary(rows),
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    run_name = safe_name(args.run_name or default_run_name())

    videos_dir = require_dir(resolve_repo_path(args.videos_dir), "videos-dir")
    perception_dir = require_dir(resolve_repo_path(args.perception_dir), "perception-dir")
    analytic_config = require_file(resolve_repo_path(args.analytic_config), "analytic-config")
    yolo_model = require_file(
        perception_host_path(args.yolo_model, perception_dir),
        "yolo-model",
    )

    work_base = perception_dir / "model_testing" / "pipeline_runs"
    output_base = ROOT / "outputs" / "pipeline_runs"
    web_app_base = ROOT / "web_app"
    web_app_runs_base = web_app_base / "outputs" / "runs"
    work_dir = resolve_repo_path(args.work_dir) if args.work_dir else work_base / run_name
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else output_base / run_name

    work_dir = prepare_clean_dir(
        work_dir,
        [work_base, web_app_runs_base],
        args.overwrite,
        "work-dir",
    )
    output_dir = prepare_clean_dir(
        output_dir,
        [output_base, web_app_runs_base],
        args.overwrite,
        "output-dir",
    )

    frames_dir = work_dir / "frames"
    masks_dir = work_dir / "yolo_masks"
    analytic_dir = output_dir / "analytic"
    manifests_dir = output_dir / "manifests"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    analytic_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(videos_dir, recursive=args.recursive)
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]
    if not videos:
        raise FileNotFoundError(f"No se encontraron videos en: {videos_dir}")

    frame_rows, video_rows = extract_frames(
        videos=videos,
        videos_dir=videos_dir,
        frames_dir=frames_dir,
        interval_seconds=max(float(args.interval_seconds), 0.0),
        jpg_quality=args.jpg_quality,
        max_frames_per_video=args.max_frames_per_video,
    )
    if not frame_rows:
        raise RuntimeError("No se extrajo ningun frame; no se puede continuar.")

    frame_manifest = manifests_dir / "frames.csv"
    video_manifest = manifests_dir / "videos.csv"
    write_manifest_csv(
        frame_manifest,
        frame_rows,
        [
            "video_name",
            "video_path",
            "video_id",
            "frame_name",
            "frame_path",
            "frame_index",
            "timestamp_sec",
        ],
    )
    write_manifest_csv(
        video_manifest,
        video_rows,
        ["video_name", "video_path", "video_id", "fps", "total_frames", "saved_frames", "status"],
    )

    run_yolo_masks(
        perception_dir=perception_dir,
        web_app_dir=web_app_base,
        docker_service=args.docker_service,
        yolo_model=yolo_model,
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        mask_mode=args.mask_mode,
        start_docker=args.start_docker,
    )

    copy_if_exists(masks_dir / "manifest.csv", manifests_dir / "yolo_manifest.csv")
    copy_if_exists(masks_dir / "summary.json", manifests_dir / "yolo_summary.json")

    run_analytic(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        analytic_dir=analytic_dir,
        config_path=analytic_config,
    )

    features_csv = analytic_dir / "tables" / "features.csv"
    feature_rows = read_csv_rows(features_csv)
    analytic_config_data = load_yaml_config(analytic_config)
    video_summary_rows = build_video_summary(
        frame_manifest_rows=frame_rows,
        video_manifest_rows=video_rows,
        feature_rows=feature_rows,
        analytic_config=analytic_config_data,
        min_valid_ratio=max(min(float(args.video_min_valid_ratio), 1.0), 0.0),
        majority_ratio=max(min(float(args.video_majority_ratio), 1.0), 0.0),
    )
    video_summary_csv = output_dir / "video_summary.csv"
    write_manifest_csv(video_summary_csv, video_summary_rows, VIDEO_SUMMARY_COLUMNS)
    indicator_summary_rows_data, indicator_distribution_rows_data = build_indicator_outputs(
        frame_manifest_rows=frame_rows,
        video_manifest_rows=video_rows,
        feature_rows=feature_rows,
        run_name=run_name,
    )
    statistics_dir = output_dir / "statistics"
    indicator_summary_csv = statistics_dir / "indicator_summary.csv"
    indicator_distributions_csv = statistics_dir / "indicator_distributions.csv"
    write_manifest_csv(
        indicator_summary_csv,
        indicator_summary_rows_data,
        INDICATOR_SUMMARY_COLUMNS,
    )
    write_manifest_csv(
        indicator_distributions_csv,
        indicator_distribution_rows_data,
        INDICATOR_DISTRIBUTION_COLUMNS,
    )
    verdict_counts = Counter(row["final_verdict"] for row in video_summary_rows)
    yolo_summary = load_json_dict(masks_dir / "summary.json")
    yolo_timings = yolo_summary.get("timings", {})
    if not isinstance(yolo_timings, dict):
        yolo_timings = {}
    summary = {
        "run_name": run_name,
        "videos_dir": str(videos_dir),
        "work_dir": str(work_dir),
        "output_dir": str(output_dir),
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "analytic_dir": str(analytic_dir),
        "video_count": len(videos),
        "frame_count": len(frame_rows),
        "yolo_model": str(yolo_model),
        "conf": args.conf,
        "device": args.device,
        "mask_mode": args.mask_mode,
        "interval_seconds": max(float(args.interval_seconds), 0.0),
        "analytic_rules": {
            "threshold_peak": float(
                nested_value(analytic_config_data, ("rules", "threshold_peak"), 0.06)
            ),
            "threshold_area": float(
                nested_value(analytic_config_data, ("rules", "threshold_area"), 0.07)
            ),
            "peak_vote_weight": float(
                nested_value(
                    analytic_config_data,
                    ("rules", "peak_vote_weight"),
                    0.5,
                )
            ),
            "area_vote_weight": float(
                nested_value(
                    analytic_config_data,
                    ("rules", "area_vote_weight"),
                    0.5,
                )
            ),
        },
        "timings": {
            "yolo_inference_total_ms": yolo_timings.get("inference_total_ms"),
            "yolo_inference_mean_ms_per_image": yolo_timings.get(
                "inference_mean_ms_per_image"
            ),
            "yolo_prediction_wall_seconds": yolo_timings.get(
                "prediction_wall_seconds"
            ),
            "yolo_images_per_second": yolo_timings.get("images_per_second"),
        },
        "features_csv": str(features_csv),
        "video_summary_csv": str(video_summary_csv),
        "indicator_summary_csv": str(indicator_summary_csv),
        "indicator_distributions_csv": str(indicator_distributions_csv),
        "video_verdict_counts": dict(verdict_counts),
        "feature_summary": summarize_features(features_csv),
    }
    summary_path = output_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nPipeline completado")
    print(f"Frames: {frames_dir}")
    print(f"Mascaras YOLO: {masks_dir}")
    print(f"Resultados analitico: {analytic_dir}")
    print(f"Veredictos por video: {video_summary_csv}")
    print(f"Resumen de indicadores: {indicator_summary_csv}")
    print(f"Distribuciones de indicadores: {indicator_distributions_csv}")
    if yolo_timings.get("inference_mean_ms_per_image") is not None:
        print(
            "Inferencia YOLO: "
            f"{float(yolo_timings['inference_mean_ms_per_image']):.3f} ms/frame"
        )
    print(f"Resumen: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
