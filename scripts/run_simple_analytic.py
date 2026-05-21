from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cone_load.features import AnalyticResult, EXTRA_COLUMNS, FEATURE_COLUMNS, empty_feature_row
from cone_load.io import (
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_MASK_EXTENSIONS,
    build_stem_index,
    ensure_dir,
    list_files,
    load_config,
    read_image_bgr,
    read_mask_rgb,
    validate_existing_dir,
)
from cone_load.masks import clean_load_mask, rgb_mask_to_binary
from cone_load.features import analyze_binary_mask
from cone_load.visualization import render_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analitico geometrico simple para mascaras CVAT de carga visible; "
            "clasifica la orientacion visual interna como centrado, izquierda o derecha."
        )
    )
    parser.add_argument("--images-dir", required=True, help="Directorio JPEGImages.")
    parser.add_argument("--masks-dir", required=True, help="Directorio SegmentationClass.")
    parser.add_argument("--output-dir", required=True, help="Directorio de salida.")
    parser.add_argument("--config", required=True, help="Archivo analytic_config.yaml.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Buscar imagenes y mascaras recursivamente.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesar solo las primeras N imagenes. Util para pruebas rapidas.",
    )
    return parser.parse_args()


def _safe_overlay_name(image_path: Path, images_dir: Path) -> str:
    try:
        rel = image_path.relative_to(images_dir)
        stem = "__".join(rel.with_suffix("").parts)
    except ValueError:
        stem = image_path.stem
    return f"{stem}_overlay.jpg"


def _row_for_processing_error(
    image_name: str,
    reason: str,
    mask_path: str = "",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    row = empty_feature_row(image_name, reason, mask_path=mask_path, warnings=warnings)
    return row


def write_features_csv(rows: list[dict[str, Any]], columns: list[str], csv_path: Path) -> int:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(io.StringIO()):
                import pandas as pd

        table = pd.DataFrame(rows).reindex(columns=columns)
        table.to_csv(csv_path, index=False)
        return int(table["is_valid"].fillna(False).astype(bool).sum())
    except Exception as exc:
        print(
            "Advertencia: pandas no pudo escribir el CSV; "
            f"se usara csv estandar ({type(exc).__name__}).",
            file=sys.stderr,
        )
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
        return sum(1 for row in rows if bool(row.get("is_valid", False)))


def process_one_image(
    image_path: Path,
    images_dir: Path,
    mask_index: dict[str, Path],
    overlay_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    image_name = image_path.name
    image_bgr = read_image_bgr(image_path)
    overlay_path = overlay_dir / _safe_overlay_name(image_path, images_dir)

    mask_path = mask_index.get(image_path.stem)
    if mask_path is None:
        row = _row_for_processing_error(image_name, "imagen_sin_mascara_correspondiente")
        result = AnalyticResult(row)
        render_overlay(image_bgr, None, result, overlay_path, config)
        row["overlay_path"] = str(overlay_path)
        return row

    warnings: list[str] = []
    binary_mask = None
    try:
        mask_rgb = read_mask_rgb(mask_path)
        if mask_rgb.shape[:2] != image_bgr.shape[:2]:
            row = _row_for_processing_error(
                image_name,
                "tamano_imagen_mascara_no_coincide",
                mask_path=str(mask_path),
            )
            result = AnalyticResult(row)
            render_overlay(image_bgr, None, result, overlay_path, config)
            row["overlay_path"] = str(overlay_path)
            return row

        conversion = rgb_mask_to_binary(mask_rgb, config)
        warnings.extend(conversion.warnings)
        clean_mask, clean_warnings = clean_load_mask(conversion.binary, config)
        warnings.extend(clean_warnings)
        binary_mask = clean_mask

        result = analyze_binary_mask(
            image_name=image_name,
            binary_mask=clean_mask,
            config=config,
            mask_path=str(mask_path),
            warnings=warnings,
            unexpected_pixel_count=conversion.unexpected_pixel_count,
            unexpected_color_count=conversion.unexpected_color_count,
        )
    except Exception as exc:
        row = _row_for_processing_error(
            image_name,
            f"error_procesando_mascara:{type(exc).__name__}",
            mask_path=str(mask_path),
            warnings=[str(exc)],
        )
        result = AnalyticResult(row)

    render_overlay(image_bgr, binary_mask, result, overlay_path, config)
    result.features["overlay_path"] = str(overlay_path)
    return result.features


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    images_dir = validate_existing_dir(args.images_dir, "images-dir")
    masks_dir = validate_existing_dir(args.masks_dir, "masks-dir")
    output_dir = ensure_dir(args.output_dir)
    overlay_dir = ensure_dir(output_dir / "overlays")
    tables_dir = ensure_dir(output_dir / "tables")

    image_exts = tuple(config.get("io", {}).get("image_extensions", DEFAULT_IMAGE_EXTENSIONS))
    mask_exts = tuple(config.get("io", {}).get("mask_extensions", DEFAULT_MASK_EXTENSIONS))

    images = list_files(images_dir, image_exts, recursive=args.recursive)
    masks = list_files(masks_dir, mask_exts, recursive=args.recursive)
    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        raise FileNotFoundError(f"No se encontraron imagenes en: {images_dir}")
    if not masks:
        raise FileNotFoundError(f"No se encontraron mascaras en: {masks_dir}")

    mask_index, duplicates = build_stem_index(masks)
    if duplicates:
        print(
            f"Advertencia: {len(duplicates)} stems de mascara duplicados. "
            "Se usara la primera ruta encontrada por stem.",
            file=sys.stderr,
        )

    rows = []
    for idx, image_path in enumerate(images, start=1):
        try:
            row = process_one_image(image_path, images_dir, mask_index, overlay_dir, config)
        except Exception as exc:
            row = _row_for_processing_error(
                image_path.name,
                f"error_procesando_imagen:{type(exc).__name__}",
                warnings=[str(exc)],
            )
        rows.append(row)
        if idx % 25 == 0 or idx == len(images):
            print(f"Procesadas {idx}/{len(images)} imagenes")

    csv_path = tables_dir / "features.csv"
    columns = FEATURE_COLUMNS + EXTRA_COLUMNS
    valid_count = write_features_csv(rows, columns, csv_path)

    print(f"CSV: {csv_path}")
    print(f"Overlays: {overlay_dir}")
    print(f"Imagenes validas: {valid_count}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
