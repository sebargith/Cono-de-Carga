from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cone_load.statistics import (  # noqa: E402
    indicator_distribution_rows,
    indicator_summary_rows,
    summarize_indicators,
)

EXPECTED_CLASSES = ("centrado", "izquierda", "derecha")
PREDICTED_CLASSES = (*EXPECTED_CLASSES, "no_evaluable")
DISPLAY_NAMES = {
    "centrado": "Centrado",
    "izquierda": "Izquierda",
    "derecha": "Derecha",
    "no_evaluable": "No evaluable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalua los resultados del analitico contra clases definidas por "
            "las carpetas Centrado, Izquierda y Derecha."
        )
    )
    parser.add_argument(
        "--features-csv",
        required=True,
        help="CSV features_with_originals.csv generado por la corrida.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde se guardara la evaluacion y el reporte.",
    )
    parser.add_argument(
        "--yolo-summary",
        default=None,
        help="summary.json de YOLO para incorporar tiempos de inferencia.",
    )
    parser.add_argument("--threshold-peak", type=float, default=0.06)
    parser.add_argument("--threshold-area", type=float, default=0.07)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reemplaza la evaluacion previa si ya existe.",
    )
    return parser.parse_args()


def normalize_class(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"centrado", "regular_simetrico", "carga_centrada"}:
        return "centrado"
    if text in {"izquierda", "asimetrico_izquierda", "tendencia_izquierda"}:
        return "izquierda"
    if text in {"derecha", "asimetrico_derecha", "tendencia_derecha"}:
        return "derecha"
    return "no_evaluable"


def expected_class_from_path(path: Path) -> str:
    for part in reversed(path.parts[:-1]):
        normalized = normalize_class(part)
        if normalized in EXPECTED_CLASSES:
            return normalized
    raise ValueError(f"No se encontro clase esperada en la ruta: {path}")


def bool_from_csv(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "si", "yes"}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"No se pudo codificar la imagen: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def read_rows(features_csv: Path) -> list[dict[str, str]]:
    with features_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"El directorio ya existe: {output_dir}. Usa --overwrite para reemplazarlo."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def copy_outputs(rows: list[dict[str, str]], output_dir: Path) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        original_path = Path(row["original_image_path"])
        expected = expected_class_from_path(original_path)
        predicted = normalize_class(row.get("predicted_class"))
        is_valid = bool_from_csv(row.get("is_valid"))
        is_correct = is_valid and predicted == expected
        class_dir = output_dir / "por_clase_esperada" / DISPLAY_NAMES[expected]
        originals_dir = class_dir / "imagenes_originales"
        overlays_dir = class_dir / "overlays_analitico"
        originals_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        if original_path.exists():
            shutil.copy2(original_path, originals_dir / original_path.name)

        overlay_raw = str(row.get("overlay_path", "")).strip()
        overlay_path = Path(overlay_raw) if overlay_raw else None
        overlay_target = overlays_dir / f"{original_path.stem}_overlay.jpg"
        if overlay_path is not None and overlay_path.exists():
            shutil.copy2(overlay_path, overlay_target)

        evaluated.append(
            {
                "image_name": original_path.name,
                "original_image_path": str(original_path),
                "expected_class": expected,
                "predicted_class": predicted,
                "is_valid": is_valid,
                "is_correct": is_correct,
                "error_reason": row.get("error_reason", ""),
                "peak_x_norm": row.get("peak_x_norm", ""),
                "area_symmetry": row.get("area_symmetry", ""),
                "centroid_x_norm": row.get("centroid_x_norm", ""),
                "profile_roughness": row.get("profile_roughness", ""),
                "overlay_path": str(overlay_target) if overlay_target.exists() else "",
            }
        )
    return evaluated


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid = sum(bool(row["is_valid"]) for row in rows)
    correct = sum(bool(row["is_correct"]) for row in rows)
    confusion = {
        expected: {
            predicted: sum(
                1
                for row in rows
                if row["expected_class"] == expected and row["predicted_class"] == predicted
            )
            for predicted in PREDICTED_CLASSES
        }
        for expected in EXPECTED_CLASSES
    }
    per_class: dict[str, Any] = {}
    for expected in EXPECTED_CLASSES:
        selected = [row for row in rows if row["expected_class"] == expected]
        selected_valid = [row for row in selected if row["is_valid"]]
        selected_correct = [row for row in selected if row["is_correct"]]
        per_class[expected] = {
            "total": len(selected),
            "evaluable": len(selected_valid),
            "correct": len(selected_correct),
            "accuracy_total": len(selected_correct) / len(selected) if selected else 0.0,
            "accuracy_evaluable": (
                len(selected_correct) / len(selected_valid) if selected_valid else 0.0
            ),
            "indicator_statistics": json_safe_indicator_summary(selected),
        }
    return {
        "total": total,
        "evaluable": valid,
        "no_evaluable": total - valid,
        "correct": correct,
        "accuracy_total": correct / total if total else 0.0,
        "accuracy_evaluable": correct / valid if valid else 0.0,
        "evaluable_ratio": valid / total if total else 0.0,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "prediction_counts": dict(Counter(row["predicted_class"] for row in rows)),
        "indicator_statistics": json_safe_indicator_summary(rows),
    }


def json_safe_indicator_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries = summarize_indicators(rows)
    output: dict[str, dict[str, Any]] = {}
    for indicator, summary in summaries.items():
        output[indicator] = {}
        for key, value in summary.items():
            if isinstance(value, float) and not math.isfinite(value):
                output[indicator][key] = None
            else:
                output[indicator][key] = value
    return output


def build_indicator_tables(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = indicator_summary_rows(
        rows,
        scope_type="prueba_tecnica",
        scope_id="total",
        scope_name="Total",
    )
    distribution_rows = indicator_distribution_rows(
        rows,
        scope_type="prueba_tecnica",
        scope_id="total",
        scope_name="Total",
    )
    for expected in EXPECTED_CLASSES:
        selected = [row for row in rows if row["expected_class"] == expected]
        summary_rows.extend(
            indicator_summary_rows(
                selected,
                scope_type="clase_esperada",
                scope_id=expected,
                scope_name=DISPLAY_NAMES[expected],
            )
        )
        distribution_rows.extend(
            indicator_distribution_rows(
                selected,
                scope_type="clase_esperada",
                scope_id=expected,
                scope_name=DISPLAY_NAMES[expected],
            )
        )
    return summary_rows, distribution_rows


def centered_text(
    canvas: np.ndarray,
    text: str,
    center_x: int,
    baseline_y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        canvas,
        text,
        (center_x - size[0] // 2, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def indicator_stat_text(metrics: dict[str, Any], expected: str, indicator: str) -> str:
    stats = (
        metrics.get("per_class", {})
        .get(expected, {})
        .get("indicator_statistics", {})
        .get(indicator, {})
    )
    median = stats.get("median")
    if median is None:
        return "sin datos"
    return f"{float(median):.3f}"


def timing_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get("timings", {}).get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def draw_report(metrics: dict[str, Any], output_path: Path) -> None:
    width, height = 1600, 1650
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    dark = (35, 35, 35)
    muted = (100, 100, 100)
    green = (65, 145, 70)
    red = (65, 75, 205)
    blue = (190, 115, 45)

    cv2.putText(
        canvas,
        "Prueba tecnica - Cono de Carga",
        (70, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        dark,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Comparacion entre seleccion visual y resultado del pipeline",
        (70, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        muted,
        2,
        cv2.LINE_AA,
    )

    cards = [
        ("Imagenes", f"{metrics['total']}", dark),
        ("Evaluables", f"{metrics['evaluable_ratio'] * 100:.1f}%", blue),
        ("Acierto total", f"{metrics['accuracy_total'] * 100:.1f}%", green),
        ("Acierto evaluables", f"{metrics['accuracy_evaluable'] * 100:.1f}%", green),
    ]
    card_y, card_h, card_gap = 150, 125, 24
    card_w = (width - 140 - card_gap * 3) // 4
    for idx, (label, value, color) in enumerate(cards):
        x = 70 + idx * (card_w + card_gap)
        cv2.rectangle(canvas, (x, card_y), (x + card_w, card_y + card_h), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x, card_y), (x + card_w, card_y + card_h), (220, 220, 220), 1)
        cv2.putText(canvas, label, (x + 20, card_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, muted, 1)
        cv2.putText(canvas, value, (x + 20, card_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, 3)

    cv2.putText(canvas, "Matriz de confusion", (70, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.9, dark, 2)
    cv2.putText(
        canvas,
        "Filas: clase esperada | Columnas: prediccion",
        (70, 375),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        muted,
        1,
    )
    matrix_x, matrix_y = 250, 420
    cell_w, cell_h = 190, 95
    for col, predicted in enumerate(PREDICTED_CLASSES):
        centered_text(
            canvas,
            DISPLAY_NAMES[predicted],
            matrix_x + col * cell_w + cell_w // 2,
            matrix_y - 22,
            0.52,
            dark,
            1,
        )
    max_value = max(
        metrics["confusion_matrix"][expected][predicted]
        for expected in EXPECTED_CLASSES
        for predicted in PREDICTED_CLASSES
    )
    for row_idx, expected in enumerate(EXPECTED_CLASSES):
        y = matrix_y + row_idx * cell_h
        cv2.putText(
            canvas,
            DISPLAY_NAMES[expected],
            (70, y + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            dark,
            2,
        )
        for col_idx, predicted in enumerate(PREDICTED_CLASSES):
            x = matrix_x + col_idx * cell_w
            value = metrics["confusion_matrix"][expected][predicted]
            intensity = value / max(max_value, 1)
            if expected == predicted:
                color = (235 - int(55 * intensity), 250 - int(30 * intensity), 235 - int(65 * intensity))
            elif value:
                color = (245 - int(20 * intensity), 240 - int(55 * intensity), 255 - int(25 * intensity))
            else:
                color = (255, 255, 255)
            cv2.rectangle(canvas, (x, y), (x + cell_w, y + cell_h), color, -1)
            cv2.rectangle(canvas, (x, y), (x + cell_w, y + cell_h), (215, 215, 215), 1)
            centered_text(canvas, str(value), x + cell_w // 2, y + 62, 0.9, dark, 2)

    chart_x, chart_y = 1080, 410
    chart_w, chart_h = 420, 310
    cv2.putText(canvas, "Acierto por clase", (1080, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.9, dark, 2)
    cv2.line(canvas, (chart_x, chart_y + chart_h), (chart_x + chart_w, chart_y + chart_h), dark, 2)
    bar_w = 90
    for idx, expected in enumerate(EXPECTED_CLASSES):
        accuracy = metrics["per_class"][expected]["accuracy_total"]
        x = chart_x + 35 + idx * 135
        bar_h = int(chart_h * accuracy)
        y = chart_y + chart_h - bar_h
        cv2.rectangle(canvas, (x, y), (x + bar_w, chart_y + chart_h), green, -1)
        centered_text(canvas, f"{accuracy * 100:.1f}%", x + bar_w // 2, y - 12, 0.52, dark, 1)
        centered_text(
            canvas,
            DISPLAY_NAMES[expected],
            x + bar_w // 2,
            chart_y + chart_h + 32,
            0.47,
            dark,
            1,
        )

    cv2.rectangle(canvas, (70, 795), (1530, 980), (255, 255, 255), -1)
    cv2.rectangle(canvas, (70, 795), (1530, 980), (220, 220, 220), 1)
    cv2.putText(canvas, "Resumen por clase", (95, 835), cv2.FONT_HERSHEY_SIMPLEX, 0.72, dark, 2)
    y = 880
    for expected in EXPECTED_CLASSES:
        item = metrics["per_class"][expected]
        text = (
            f"{DISPLAY_NAMES[expected]}: {item['correct']}/{item['total']} correctas "
            f"| {item['evaluable']}/{item['total']} evaluables"
        )
        cv2.putText(canvas, text, (105, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, dark, 2)
        y += 40
    cv2.putText(
        canvas,
        "Acierto total considera las imagenes no evaluables como error.",
        (930, 950),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        red,
        1,
    )

    cv2.putText(
        canvas,
        "Indicadores geometricos por clase esperada",
        (70, 1055),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        dark,
        2,
    )
    cv2.putText(
        canvas,
        "Cada valor es la mediana de los frames validos de la clase.",
        (70, 1090),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        muted,
        1,
    )

    table_x, table_y = 70, 1115
    row_label_w, indicator_w = 220, 430
    header_h, row_h = 55, 72
    headers = (
        ("Clase esperada", row_label_w),
        ("Pico mediano", indicator_w),
        ("Simetria mediana", indicator_w),
        ("Centroide mediano", indicator_w),
    )
    current_x = table_x
    for header, column_width in headers:
        cv2.rectangle(
            canvas,
            (current_x, table_y),
            (current_x + column_width, table_y + header_h),
            (235, 239, 241),
            -1,
        )
        cv2.rectangle(
            canvas,
            (current_x, table_y),
            (current_x + column_width, table_y + header_h),
            (205, 210, 212),
            1,
        )
        cv2.putText(
            canvas,
            header,
            (current_x + 14, table_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            dark,
            1,
        )
        current_x += column_width

    indicator_names = ("peak_x_norm", "area_symmetry", "centroid_x_norm")
    for row_index, expected in enumerate(EXPECTED_CLASSES):
        y1 = table_y + header_h + row_index * row_h
        current_x = table_x
        values = [DISPLAY_NAMES[expected]]
        values.extend(
            indicator_stat_text(metrics, expected, indicator)
            for indicator in indicator_names
        )
        for column_index, ((_, column_width), value) in enumerate(
            zip(headers, values)
        ):
            fill = (255, 255, 255) if row_index % 2 == 0 else (248, 249, 250)
            cv2.rectangle(
                canvas,
                (current_x, y1),
                (current_x + column_width, y1 + row_h),
                fill,
                -1,
            )
            cv2.rectangle(
                canvas,
                (current_x, y1),
                (current_x + column_width, y1 + row_h),
                (215, 218, 220),
                1,
            )
            cv2.putText(
                canvas,
                value,
                (current_x + 14, y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55 if column_index else 0.6,
                dark,
                2 if column_index == 0 else 1,
            )
            current_x += column_width

    threshold_peak = float(metrics.get("thresholds", {}).get("peak", 0.06))
    threshold_area = float(metrics.get("thresholds", {}).get("area", 0.07))
    cv2.putText(
        canvas,
        (
            f"Referencias: pico < -{threshold_peak:.2f} izquierda y "
            f"> {threshold_peak:.2f} derecha | simetria < -{threshold_area:.2f} "
            f"izquierda y > {threshold_area:.2f} derecha."
        ),
        (70, 1405),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        muted,
        1,
    )
    cv2.putText(
        canvas,
        "El centroide relativo es descriptivo: negativo izquierda, cero centro, positivo derecha.",
        (70, 1432),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        muted,
        1,
    )

    cv2.putText(
        canvas,
        "Rendimiento de percepcion",
        (70, 1470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        dark,
        2,
    )
    timing_cards = [
        (
            "Inferencia YOLO por imagen",
            timing_value(metrics, "inference_mean_ms_per_image"),
            "ms",
        ),
        (
            "Inferencia YOLO acumulada",
            timing_value(metrics, "inference_total_ms"),
            "ms",
        ),
        (
            "Generacion completa de mascaras",
            timing_value(metrics, "prediction_wall_seconds"),
            "s",
        ),
    ]
    timing_y, timing_h, timing_gap = 1495, 110, 24
    timing_w = (width - 140 - timing_gap * 2) // 3
    for index, (label, value, unit) in enumerate(timing_cards):
        x = 70 + index * (timing_w + timing_gap)
        cv2.rectangle(
            canvas,
            (x, timing_y),
            (x + timing_w, timing_y + timing_h),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            canvas,
            (x, timing_y),
            (x + timing_w, timing_y + timing_h),
            (220, 220, 220),
            1,
        )
        cv2.putText(
            canvas,
            label,
            (x + 18, timing_y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            muted,
            1,
        )
        value_text = "no disponible" if value is None else f"{value:.1f} {unit}"
        cv2.putText(
            canvas,
            value_text,
            (x + 18, timing_y + 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            blue,
            2,
        )
    write_image(output_path, canvas)


def main() -> int:
    args = parse_args()
    features_csv = Path(args.features_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not features_csv.is_file():
        raise FileNotFoundError(f"No existe features-csv: {features_csv}")
    prepare_output_dir(output_dir, args.overwrite)

    rows = read_rows(features_csv)
    if not rows:
        raise ValueError(f"El CSV no contiene resultados: {features_csv}")
    evaluated = copy_outputs(rows, output_dir)
    metrics = build_metrics(evaluated)
    metrics["thresholds"] = {
        "peak": float(args.threshold_peak),
        "area": float(args.threshold_area),
    }
    metrics["timings"] = {}
    if args.yolo_summary:
        yolo_summary_path = Path(args.yolo_summary).resolve()
        if not yolo_summary_path.is_file():
            raise FileNotFoundError(
                f"No existe yolo-summary: {yolo_summary_path}"
            )
        yolo_summary = read_json_dict(yolo_summary_path)
        yolo_timings = yolo_summary.get("timings", {})
        if isinstance(yolo_timings, dict):
            metrics["timings"] = yolo_timings

    write_csv(
        output_dir / "evaluacion_por_imagen.csv",
        evaluated,
        [
            "image_name",
            "original_image_path",
            "expected_class",
            "predicted_class",
            "is_valid",
            "is_correct",
            "error_reason",
            "peak_x_norm",
            "area_symmetry",
            "centroid_x_norm",
            "profile_roughness",
            "overlay_path",
        ],
    )
    indicator_rows, distribution_rows = build_indicator_tables(evaluated)
    write_csv(
        output_dir / "indicadores_geometricos.csv",
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
    write_csv(
        output_dir / "distribuciones_geometricas.csv",
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
    (output_dir / "resumen_evaluacion.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path = output_dir / "reporte_prueba_tecnica.png"
    draw_report(metrics, report_path)

    print("Evaluacion completada")
    print(f"Imagenes: {metrics['total']}")
    print(f"Evaluables: {metrics['evaluable']}/{metrics['total']}")
    print(f"Correctas: {metrics['correct']}/{metrics['total']}")
    print(f"Acierto total: {metrics['accuracy_total'] * 100:.2f}%")
    print(f"Acierto evaluables: {metrics['accuracy_evaluable'] * 100:.2f}%")
    for expected in EXPECTED_CLASSES:
        item = metrics["per_class"][expected]
        print(
            f"{DISPLAY_NAMES[expected]}: {item['correct']}/{item['total']} "
            f"({item['accuracy_total'] * 100:.2f}%)"
        )
    print(f"Reporte: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
