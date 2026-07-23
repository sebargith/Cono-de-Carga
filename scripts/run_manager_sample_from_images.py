from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cone_load.statistics import (  # noqa: E402
    indicator_distribution_rows,
    indicator_summary_rows,
)

from run_pipeline_from_videos import (  # noqa: E402
    DEFAULT_MODEL,
    bool_from_csv,
    finite_float_from_csv,
    fmt_float,
    is_relative_to,
    load_yaml_config,
    nested_value,
    json_safe_indicator_summary,
    perception_host_path,
    prepare_clean_dir,
    read_csv_rows,
    require_dir,
    require_file,
    resolve_repo_path,
    run_analytic,
    run_yolo_masks,
    safe_name,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
RESULT_CLASSES = ("izquierda", "centrado", "derecha", "no_evaluable")


def write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise OSError(f"No se pudo codificar imagen: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def parse_args() -> argparse.Namespace:
    default_images = ROOT / "Presentación Psinet" / "data"
    parser = argparse.ArgumentParser(
        description=(
            "Corre YOLO + analitico sobre una muestra de imagenes sueltas, "
            "genera histogramas y separa resultados por izquierda/centrado/derecha."
        )
    )
    parser.add_argument(
        "--images-dir",
        default=str(default_images),
        help="Carpeta con imagenes de entrada. Por defecto usa Presentacion Psinet/data.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre de la corrida. Por defecto usa timestamp.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Carpeta temporal visible para Docker. Por defecto queda en "
            "perception/model_testing/presentation_sample_runs/<run-name>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Carpeta final de resultados. Por defecto queda en "
            "Presentacion Psinet/resultados_muestra/<run-name>."
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
    parser.add_argument("--recursive", action="store_true", help="Buscar imagenes recursivamente.")
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo las primeras N.")
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
        help="Borra una corrida previa con el mismo nombre.",
    )
    return parser.parse_args()


def default_run_name() -> str:
    return datetime.now().strftime("muestra_%Y%m%d_%H%M%S")


def list_images(images_dir: Path, recursive: bool) -> list[Path]:
    iterator = images_dir.rglob("*") if recursive else images_dir.glob("*")
    allowed = {ext.lower() for ext in IMAGE_EXTENSIONS}
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in allowed)


def unique_staged_name(image_path: Path, images_dir: Path, used: set[str]) -> str:
    try:
        rel = image_path.relative_to(images_dir)
        base = "__".join(rel.with_suffix("").parts)
    except ValueError:
        base = image_path.stem
    stem = safe_name(base)
    suffix = image_path.suffix.lower() or ".jpg"
    candidate = f"{stem}{suffix}"
    idx = 2
    while candidate in used:
        candidate = f"{stem}__{idx:03d}{suffix}"
        idx += 1
    used.add(candidate)
    return candidate


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stage_images(
    images: list[Path],
    images_dir: Path,
    staged_dir: Path,
) -> list[dict[str, Any]]:
    staged_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    rows: list[dict[str, Any]] = []
    for idx, image_path in enumerate(images, start=1):
        staged_name = unique_staged_name(image_path, images_dir, used)
        staged_path = staged_dir / staged_name
        shutil.copy2(image_path, staged_path)
        rows.append(
            {
                "sample_index": idx,
                "image_name": staged_name,
                "original_image_name": image_path.name,
                "original_image_path": str(image_path),
                "staged_image_path": str(staged_path),
            }
        )
    return rows


def normalize_result_class(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"izquierda", "asimetrico_izquierda", "tendencia_izquierda"}:
        return "izquierda"
    if text in {"derecha", "asimetrico_derecha", "tendencia_derecha"}:
        return "derecha"
    if text in {"centrado", "regular_simetrico", "carga_centrada"}:
        return "centrado"
    return "no_evaluable"


def class_counts_from_rows(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[normalize_result_class(row.get("predicted_class"))] += 1
    for class_name in RESULT_CLASSES:
        counts.setdefault(class_name, 0)
    return counts


def copy_classified_outputs(
    feature_rows: list[dict[str, Any]],
    manifest_by_image: dict[str, dict[str, Any]],
    separated_dir: Path,
) -> list[dict[str, Any]]:
    copied_rows: list[dict[str, Any]] = []
    for class_name in RESULT_CLASSES:
        (separated_dir / class_name / "imagenes").mkdir(parents=True, exist_ok=True)
        (separated_dir / class_name / "overlays").mkdir(parents=True, exist_ok=True)

    for row in feature_rows:
        image_name = str(row.get("image_name", ""))
        class_name = normalize_result_class(row.get("predicted_class"))
        manifest = manifest_by_image.get(image_name, {})

        original_path = Path(str(manifest.get("original_image_path", "")))
        image_target = separated_dir / class_name / "imagenes" / image_name
        overlay_target = separated_dir / class_name / "overlays" / f"{Path(image_name).stem}_overlay.jpg"

        if original_path.exists():
            shutil.copy2(original_path, image_target)

        overlay_path_raw = str(row.get("overlay_path", "")).strip()
        overlay_path = Path(overlay_path_raw) if overlay_path_raw else None
        if overlay_path is not None and overlay_path.exists():
            shutil.copy2(overlay_path, overlay_target)

        copied_rows.append(
            {
                "image_name": image_name,
                "original_image_name": manifest.get("original_image_name", ""),
                "original_image_path": manifest.get("original_image_path", ""),
                "predicted_class": row.get("predicted_class", ""),
                "result_group": class_name,
                "is_valid": row.get("is_valid", ""),
                "error_reason": row.get("error_reason", ""),
                "separated_image_path": str(image_target) if image_target.exists() else "",
                "separated_overlay_path": str(overlay_target) if overlay_target.exists() else "",
            }
        )
    return copied_rows


def enhance_features_with_originals(
    feature_rows: list[dict[str, Any]],
    manifest_by_image: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enhanced: list[dict[str, Any]] = []
    for row in feature_rows:
        image_name = str(row.get("image_name", ""))
        manifest = manifest_by_image.get(image_name, {})
        item = dict(row)
        item["original_image_name"] = manifest.get("original_image_name", "")
        item["original_image_path"] = manifest.get("original_image_path", "")
        item["result_group"] = normalize_result_class(row.get("predicted_class"))
        enhanced.append(item)
    return enhanced


def ratio_text(count: int, total: int) -> str:
    return fmt_float(float(count / total) if total else 0.0)


def draw_bar_chart(
    counts: Counter[str],
    output_path: Path,
    title: str,
    total: int,
) -> None:
    width, height = 1100, 720
    margin_left, margin_right = 100, 60
    margin_top, margin_bottom = 110, 125
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    axis_color = (40, 40, 40)
    grid_color = (220, 220, 220)
    text_color = (25, 25, 25)
    palette = {
        "izquierda": (65, 110, 210),
        "centrado": (45, 150, 80),
        "derecha": (40, 170, 210),
        "no_evaluable": (130, 130, 130),
    }

    cv2.putText(canvas, title, (margin_left, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, text_color, 2)
    cv2.putText(
        canvas,
        f"Total imagenes: {total}",
        (margin_left, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (85, 85, 85),
        2,
    )

    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top
    max_count = max(max(counts.values(), default=0), 1)
    y_max = int(math.ceil(max_count / 10.0) * 10) if max_count > 10 else max_count

    for tick in range(0, 6):
        y_value = y_max * tick / 5.0
        y = int(plot_bottom - (y_value / y_max) * plot_h)
        cv2.line(canvas, (plot_left, y), (plot_right, y), grid_color, 1)
        cv2.putText(
            canvas,
            str(int(round(y_value))),
            (35, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (90, 90, 90),
            1,
        )

    cv2.line(canvas, (plot_left, plot_top), (plot_left, plot_bottom), axis_color, 2)
    cv2.line(canvas, (plot_left, plot_bottom), (plot_right, plot_bottom), axis_color, 2)

    classes = list(RESULT_CLASSES)
    slot = plot_w / len(classes)
    bar_w = int(slot * 0.52)
    for idx, class_name in enumerate(classes):
        count = int(counts[class_name])
        x_center = int(plot_left + slot * idx + slot / 2)
        x1 = x_center - bar_w // 2
        x2 = x_center + bar_w // 2
        bar_h = int((count / y_max) * plot_h) if y_max else 0
        y1 = plot_bottom - bar_h
        cv2.rectangle(canvas, (x1, y1), (x2, plot_bottom), palette[class_name], -1)
        cv2.rectangle(canvas, (x1, y1), (x2, plot_bottom), (60, 60, 60), 1)
        cv2.putText(
            canvas,
            str(count),
            (x_center - 18, max(y1 - 14, plot_top + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            text_color,
            2,
        )
        pct = 100.0 * count / total if total else 0.0
        cv2.putText(
            canvas,
            f"{pct:.1f}%",
            (x_center - 35, plot_bottom + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 80, 80),
            2,
        )
        cv2.putText(
            canvas,
            class_name,
            (x_center - 65, plot_bottom + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            text_color,
            2,
        )

    write_image(output_path, canvas)


def draw_numeric_histogram(
    values: list[float],
    output_path: Path,
    title: str,
    x_min: float,
    x_max: float,
    thresholds: tuple[float, float] | None = None,
) -> None:
    width, height = 1100, 720
    margin_left, margin_right = 100, 60
    margin_top, margin_bottom = 110, 110
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    axis_color = (40, 40, 40)
    grid_color = (220, 220, 220)
    text_color = (25, 25, 25)
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    cv2.putText(canvas, title, (margin_left, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, text_color, 2)
    cv2.putText(
        canvas,
        f"Valores validos: {len(values)}",
        (margin_left, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (85, 85, 85),
        2,
    )

    if values:
        bins = np.linspace(x_min, x_max, 31)
        hist, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    else:
        edges = np.linspace(x_min, x_max, 31)
        hist = np.zeros(30, dtype=int)

    max_count = max(int(hist.max()) if hist.size else 0, 1)
    for tick in range(0, 6):
        y_value = max_count * tick / 5.0
        y = int(plot_bottom - (y_value / max_count) * plot_h)
        cv2.line(canvas, (plot_left, y), (plot_right, y), grid_color, 1)
        cv2.putText(
            canvas,
            str(int(round(y_value))),
            (35, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (90, 90, 90),
            1,
        )

    cv2.line(canvas, (plot_left, plot_top), (plot_left, plot_bottom), axis_color, 2)
    cv2.line(canvas, (plot_left, plot_bottom), (plot_right, plot_bottom), axis_color, 2)

    bar_gap = 2
    for idx, count in enumerate(hist):
        bin_left = edges[idx]
        bin_right = edges[idx + 1]
        x1 = int(plot_left + ((bin_left - x_min) / (x_max - x_min)) * plot_w) + bar_gap
        x2 = int(plot_left + ((bin_right - x_min) / (x_max - x_min)) * plot_w) - bar_gap
        bar_h = int((int(count) / max_count) * plot_h)
        y1 = plot_bottom - bar_h
        cv2.rectangle(canvas, (x1, y1), (x2, plot_bottom), (80, 145, 210), -1)

    if thresholds is not None:
        for threshold in thresholds:
            if x_min <= threshold <= x_max:
                x = int(plot_left + ((threshold - x_min) / (x_max - x_min)) * plot_w)
                cv2.line(canvas, (x, plot_top), (x, plot_bottom), (40, 90, 220), 2)
                cv2.putText(
                    canvas,
                    f"{threshold:.2f}",
                    (x - 28, plot_top - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 90, 220),
                    2,
                )

    for value, label in ((x_min, f"{x_min:.1f}"), (0.0, "0"), (x_max, f"{x_max:.1f}")):
        if x_min <= value <= x_max:
            x = int(plot_left + ((value - x_min) / (x_max - x_min)) * plot_w)
            cv2.putText(canvas, label, (x - 22, plot_bottom + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    cv2.putText(
        canvas,
        "izquierda < 0                         derecha > 0",
        (plot_left + 240, plot_bottom + 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (85, 85, 85),
        2,
    )

    write_image(output_path, canvas)


def build_summary(
    rows: list[dict[str, Any]],
    counts: Counter[str],
    invalid_reasons: Counter[str],
    paths: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    total = len(rows)
    valid_rows = [row for row in rows if bool_from_csv(row.get("is_valid", ""))]
    valid = len(valid_rows)
    return {
        "total_images": total,
        "valid_images": valid,
        "invalid_images": total - valid,
        "valid_ratio": ratio_text(valid, total),
        "class_counts": {class_name: int(counts[class_name]) for class_name in RESULT_CLASSES},
        "class_ratios": {
            class_name: ratio_text(int(counts[class_name]), total) for class_name in RESULT_CLASSES
        },
        "invalid_reason_counts": dict(invalid_reasons),
        "indicator_statistics": json_safe_indicator_summary(rows),
        "settings": {
            "conf": args.conf,
            "imgsz": args.imgsz,
            "device": args.device,
            "mask_mode": args.mask_mode,
        },
        "paths": paths,
    }


def main() -> int:
    args = parse_args()
    run_name = safe_name(args.run_name or default_run_name())

    images_dir = require_dir(resolve_repo_path(args.images_dir), "images-dir")
    perception_dir = require_dir(resolve_repo_path(args.perception_dir), "perception-dir")
    analytic_config = require_file(resolve_repo_path(args.analytic_config), "analytic-config")
    yolo_model = require_file(perception_host_path(args.yolo_model, perception_dir), "yolo-model")

    work_base = perception_dir / "model_testing" / "presentation_sample_runs"
    output_base = ROOT / "Presentación Psinet" / "resultados_muestra"
    work_dir = resolve_repo_path(args.work_dir) if args.work_dir else work_base / run_name
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else output_base / run_name

    work_dir = prepare_clean_dir(work_dir, [work_base], args.overwrite, "work-dir")
    output_dir = prepare_clean_dir(output_dir, [output_base, ROOT / "outputs"], args.overwrite, "output-dir")

    staged_images_dir = work_dir / "images"
    masks_dir = work_dir / "yolo_masks"
    analytic_dir = output_dir / "analytic"
    manifests_dir = output_dir / "manifests"
    summary_dir = output_dir / "summary"
    separated_dir = output_dir / "separados_por_resultado"
    staged_images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    analytic_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir, recursive=args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No se encontraron imagenes en: {images_dir}")

    manifest_rows = stage_images(images, images_dir, staged_images_dir)
    write_csv(
        manifests_dir / "images.csv",
        manifest_rows,
        [
            "sample_index",
            "image_name",
            "original_image_name",
            "original_image_path",
            "staged_image_path",
        ],
    )
    print(f"Imagenes preparadas: {len(manifest_rows)}")
    print(f"Work dir: {work_dir}")

    run_yolo_masks(
        perception_dir=perception_dir,
        web_app_dir=None,
        docker_service=args.docker_service,
        yolo_model=yolo_model,
        frames_dir=staged_images_dir,
        masks_dir=masks_dir,
        conf=float(args.conf),
        imgsz=args.imgsz,
        device=str(args.device),
        mask_mode=args.mask_mode,
        start_docker=bool(args.start_docker),
    )
    run_analytic(
        frames_dir=staged_images_dir,
        masks_dir=masks_dir,
        analytic_dir=analytic_dir,
        config_path=analytic_config,
    )

    features_csv = analytic_dir / "tables" / "features.csv"
    feature_rows = read_csv_rows(features_csv)
    manifest_by_image = {str(row["image_name"]): row for row in manifest_rows}
    enhanced_rows = enhance_features_with_originals(feature_rows, manifest_by_image)
    enhanced_columns = (
        ["original_image_name", "original_image_path", "result_group"]
        + list(feature_rows[0].keys() if feature_rows else [])
    )
    write_csv(analytic_dir / "tables" / "features_with_originals.csv", enhanced_rows, enhanced_columns)

    copied_rows = copy_classified_outputs(feature_rows, manifest_by_image, separated_dir)
    write_csv(
        summary_dir / "imagenes_separadas.csv",
        copied_rows,
        [
            "image_name",
            "original_image_name",
            "original_image_path",
            "predicted_class",
            "result_group",
            "is_valid",
            "error_reason",
            "separated_image_path",
            "separated_overlay_path",
        ],
    )

    counts = class_counts_from_rows(feature_rows)
    invalid_reasons: Counter[str] = Counter(
        str(row.get("error_reason", "") or "sin_error_reason")
        for row in feature_rows
        if not bool_from_csv(row.get("is_valid", ""))
    )
    count_rows = []
    total = len(feature_rows)
    for class_name in RESULT_CLASSES:
        count = int(counts[class_name])
        count_rows.append(
            {
                "result_group": class_name,
                "count": count,
                "ratio_total": ratio_text(count, total),
                "imagenes_dir": str(separated_dir / class_name / "imagenes"),
                "overlays_dir": str(separated_dir / class_name / "overlays"),
            }
        )
    write_csv(
        summary_dir / "conteo_resultados.csv",
        count_rows,
        ["result_group", "count", "ratio_total", "imagenes_dir", "overlays_dir"],
    )

    valid_rows = [row for row in feature_rows if bool_from_csv(row.get("is_valid", ""))]
    peak_values = [
        value
        for value in (finite_float_from_csv(row.get("peak_x_norm")) for row in valid_rows)
        if value is not None
    ]
    area_values = [
        value
        for value in (finite_float_from_csv(row.get("area_symmetry")) for row in valid_rows)
        if value is not None
    ]
    centroid_values = [
        value
        for value in (
            finite_float_from_csv(row.get("centroid_x_norm")) for row in valid_rows
        )
        if value is not None
    ]
    config = load_yaml_config(analytic_config)
    threshold_peak = float(nested_value(config, ("rules", "threshold_peak"), 0.09))
    threshold_area = float(nested_value(config, ("rules", "threshold_area"), 0.10))

    draw_bar_chart(
        counts,
        summary_dir / "histograma_resultados.png",
        "Distribucion de resultados de la muestra",
        total=total,
    )
    draw_numeric_histogram(
        peak_values,
        summary_dir / "histograma_peak_x_norm.png",
        "Histograma peak_x_norm",
        x_min=-0.5,
        x_max=0.5,
        thresholds=(-threshold_peak, threshold_peak),
    )
    draw_numeric_histogram(
        area_values,
        summary_dir / "histograma_area_symmetry.png",
        "Histograma area_symmetry",
        x_min=-1.0,
        x_max=1.0,
        thresholds=(-threshold_area, threshold_area),
    )
    draw_numeric_histogram(
        centroid_values,
        summary_dir / "histograma_centroid_x_norm.png",
        "Histograma centroid_x_norm",
        x_min=-0.5,
        x_max=0.5,
        thresholds=None,
    )

    indicator_rows = indicator_summary_rows(
        feature_rows,
        scope_type="muestra",
        scope_id=run_name,
        scope_name=run_name,
    )
    for row in indicator_rows:
        for field in ("p25", "median", "p75"):
            row[field] = fmt_float(float(row[field]))
    indicators_csv = summary_dir / "indicadores_geometricos.csv"
    write_csv(
        indicators_csv,
        indicator_rows,
        [
            "scope_type",
            "scope_id",
            "scope_name",
            "indicator",
            "indicator_label",
            "count",
            "p25",
            "median",
            "p75",
        ],
    )
    distribution_rows = indicator_distribution_rows(
        feature_rows,
        scope_type="muestra",
        scope_id=run_name,
        scope_name=run_name,
    )
    for row in distribution_rows:
        row["bin_start"] = fmt_float(float(row["bin_start"]))
        row["bin_end"] = fmt_float(float(row["bin_end"]))
    distributions_csv = summary_dir / "distribuciones_geometricas.csv"
    write_csv(
        distributions_csv,
        distribution_rows,
        [
            "scope_type",
            "scope_id",
            "scope_name",
            "indicator",
            "indicator_label",
            "bin_index",
            "bin_start",
            "bin_end",
            "count",
        ],
    )

    paths = {
        "input_images_dir": str(images_dir),
        "work_dir": str(work_dir),
        "staged_images_dir": str(staged_images_dir),
        "yolo_masks_dir": str(masks_dir),
        "analytic_dir": str(analytic_dir),
        "features_csv": str(features_csv),
        "features_with_originals_csv": str(analytic_dir / "tables" / "features_with_originals.csv"),
        "class_counts_csv": str(summary_dir / "conteo_resultados.csv"),
        "separated_dir": str(separated_dir),
        "histogram_results": str(summary_dir / "histograma_resultados.png"),
        "histogram_peak_x_norm": str(summary_dir / "histograma_peak_x_norm.png"),
        "histogram_area_symmetry": str(summary_dir / "histograma_area_symmetry.png"),
        "histogram_centroid_x_norm": str(summary_dir / "histograma_centroid_x_norm.png"),
        "indicator_summary_csv": str(indicators_csv),
        "indicator_distributions_csv": str(distributions_csv),
    }
    summary = build_summary(feature_rows, counts, invalid_reasons, paths, args)
    summary_path = summary_dir / "resumen_muestra.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nMuestra procesada")
    print(f"Imagenes: {total}")
    print(f"Validas: {summary['valid_images']}/{total}")
    print(f"Izquierda: {counts['izquierda']}")
    print(f"Centrado: {counts['centrado']}")
    print(f"Derecha: {counts['derecha']}")
    print(f"No evaluable: {counts['no_evaluable']}")
    print(f"Resultados: {output_dir}")
    print(f"Resumen: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
