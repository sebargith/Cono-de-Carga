from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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
    }


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


def draw_report(metrics: dict[str, Any], output_path: Path) -> None:
    width, height = 1600, 1050
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
            "profile_roughness",
            "overlay_path",
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
