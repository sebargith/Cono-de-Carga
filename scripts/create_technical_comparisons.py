from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_NAMES = {
    "centrado": "Centrado",
    "izquierda": "Izquierda",
    "derecha": "Derecha",
    "no_evaluable": "No evaluable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea comparaciones lado a lado entre imagenes originales y overlays "
            "de una prueba tecnica etiquetada."
        )
    )
    parser.add_argument(
        "--evaluation-csv",
        required=True,
        help="Ruta a evaluacion_por_imagen.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio de salida para las comparaciones por clase esperada.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar comparaciones existentes.",
    )
    return parser.parse_args()


def bool_from_csv(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "si", "yes"}


def display_class(value: Any) -> str:
    normalized = str(value or "no_evaluable").strip().lower()
    return CLASS_NAMES.get(normalized, normalized.replace("_", " ").title())


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"No existe imagen: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"No se pudo leer imagen: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise OSError(f"No se pudo codificar imagen: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def metric_text(row: dict[str, str]) -> str:
    parts: list[str] = []
    for key, label in (
        ("peak_x_norm", "peak_x_norm"),
        ("area_symmetry", "area_symmetry"),
        ("profile_roughness", "roughness"),
    ):
        raw = str(row.get(key, "")).strip()
        try:
            parts.append(f"{label}: {float(raw):.3f}")
        except (TypeError, ValueError):
            parts.append(f"{label}: n/a")
    return "   |   ".join(parts)


def create_comparison(row: dict[str, str]) -> np.ndarray:
    original = read_image(Path(row["original_image_path"]))
    overlay = read_image(Path(row["overlay_path"]))

    width, height = 1800, 760
    margin = 35
    gap = 35
    header_h = 95
    image_h = 500
    panel_w = (width - 2 * margin - gap) // 2
    left_x = margin
    right_x = margin + panel_w + gap
    image_y = header_h

    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    expected = display_class(row.get("expected_class"))
    predicted = display_class(row.get("predicted_class"))
    is_valid = bool_from_csv(row.get("is_valid"))
    is_correct = bool_from_csv(row.get("is_correct"))
    prediction_color = (55, 145, 65) if is_correct else (55, 65, 210)
    if not is_valid:
        prediction_color = (115, 115, 115)

    draw_text(canvas, f"Esperado: {expected}", (left_x, 58), 0.88, (35, 35, 35), 2)
    draw_text(
        canvas,
        f"Prediccion: {predicted}",
        (right_x, 58),
        0.88,
        prediction_color,
        2,
    )

    left_panel = fit_image(original, panel_w, image_h)
    right_panel = fit_image(overlay, panel_w, image_h)
    canvas[image_y : image_y + image_h, left_x : left_x + panel_w] = left_panel
    canvas[image_y : image_y + image_h, right_x : right_x + panel_w] = right_panel
    cv2.rectangle(
        canvas,
        (left_x, image_y),
        (left_x + panel_w, image_y + image_h),
        (210, 210, 210),
        1,
    )
    cv2.rectangle(
        canvas,
        (right_x, image_y),
        (right_x + panel_w, image_y + image_h),
        (210, 210, 210),
        1,
    )

    draw_text(canvas, "Frame original", (left_x + 8, 635), 0.62, (95, 95, 95), 1)
    draw_text(canvas, "Resultado del analitico", (right_x + 8, 635), 0.62, (95, 95, 95), 1)
    status = "CORRECTO" if is_correct else "INCORRECTO"
    if not is_valid:
        status = "NO EVALUABLE"
    draw_text(canvas, f"Estado: {status}", (margin, 694), 0.68, prediction_color, 2)
    draw_text(canvas, metric_text(row), (505, 694), 0.55, (55, 55, 55), 1)
    draw_text(canvas, str(row.get("image_name", "")), (margin, 735), 0.48, (110, 110, 110), 1)
    return canvas


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe evaluation-csv: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    evaluation_csv = Path(args.evaluation_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = read_rows(evaluation_csv)
    if not rows:
        raise ValueError(f"El CSV no contiene resultados: {evaluation_csv}")

    counts: Counter[str] = Counter()
    for row in sorted(rows, key=lambda item: (item.get("expected_class", ""), item.get("image_name", ""))):
        expected = display_class(row.get("expected_class"))
        class_dir = output_dir / expected
        output_path = class_dir / f"{Path(row['image_name']).stem}_comparacion.jpg"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Ya existe comparacion: {output_path}. Usa --overwrite para reemplazarla."
            )
        comparison = create_comparison(row)
        write_image(output_path, comparison)
        counts[expected] += 1

    print("Comparaciones creadas")
    for class_name in sorted(counts):
        print(f"{class_name}: {counts[class_name]}")
    print(f"Total: {sum(counts.values())}")
    print(f"Directorio: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
