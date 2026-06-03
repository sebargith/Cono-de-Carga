from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cone_perception.io import IMAGE_EXTENSIONS, ensure_dir, get_nested, list_files, load_yaml


@dataclass
class MaskPrediction:
    mask: np.ndarray
    has_prediction: bool
    num_instances: int
    num_instances_used: int
    mask_area_px: int
    max_conf: float | None
    warning: str


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Corre YOLO-seg y guarda mascaras RGB compatibles con el analitico "
            "geometrico del cono de carga."
        )
    )
    parser.add_argument("--model", required=True, help="Ruta a best.pt o modelo YOLO-seg.")
    parser.add_argument("--images-dir", required=True, help="Carpeta con frames de entrada.")
    parser.add_argument(
        "--masks-dir",
        default=None,
        help="Carpeta de salida para mascaras compatibles con el analitico.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "perception_config.yaml"),
        help="Archivo de configuracion YAML.",
    )
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--class-id", type=int, default=None, help="Clase YOLO a convertir.")
    parser.add_argument(
        "--mask-mode",
        choices=("largest", "merge_all"),
        default=None,
        help="Como combinar instancias detectadas de carga.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Cantidad de imagenes por llamada a YOLO.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Buscar imagenes recursivamente dentro de images-dir.",
    )
    parser.add_argument(
        "--output-ext",
        default=".png",
        help="Extension de mascaras de salida. Se recomienda .png.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescribir mascaras y manifest existentes.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Ruta opcional del manifest CSV. Por defecto queda en masks-dir/manifest.csv.",
    )
    return parser.parse_args()


def validate_inputs(model_path: Path, images_dir: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"No existe images-dir: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"images-dir no es un directorio: {images_dir}")


def find_duplicate_stems(images: list[Path]) -> dict[str, list[Path]]:
    by_stem: dict[str, list[Path]] = {}
    for image_path in images:
        by_stem.setdefault(image_path.stem, []).append(image_path)
    return {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}


def chunked(paths: list[Path], chunk_size: int) -> list[list[Path]]:
    size = max(int(chunk_size), 1)
    return [paths[idx : idx + size] for idx in range(0, len(paths), size)]


def normalize_output_ext(extension: str) -> str:
    ext = extension.strip()
    if not ext:
        ext = ".png"
    ext = ext if ext.startswith(".") else f".{ext}"
    if ext.lower() in {".jpg", ".jpeg"}:
        raise ValueError(
            "No guardes mascaras analiticas como JPG/JPEG: la compresion altera "
            "el color RGB exacto de la clase. Usa .png, .bmp o .tif."
        )
    return ext


def result_shape(result: Any) -> tuple[int, int]:
    if getattr(result, "orig_img", None) is not None:
        return result.orig_img.shape[:2]
    if getattr(result, "orig_shape", None):
        h, w = result.orig_shape[:2]
        return int(h), int(w)
    raise ValueError("No se pudo determinar el tamano original de la imagen.")


def _result_class_ids(result: Any) -> list[int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "cls", None) is None:
        return []
    return [int(v) for v in boxes.cls.detach().cpu().numpy().tolist()]


def _result_confs(result: Any) -> list[float]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "conf", None) is None:
        return []
    return [float(v) for v in boxes.conf.detach().cpu().numpy().tolist()]


def _polygon_to_mask(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    instance_mask = np.zeros((h, w), dtype=np.uint8)
    if poly is None or len(poly) < 3:
        return instance_mask

    points = np.rint(poly).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    cv2.fillPoly(instance_mask, [points], 1)
    return instance_mask


def result_to_binary_mask(result: Any, class_id: int, mask_mode: str) -> MaskPrediction:
    h, w = result_shape(result)
    empty = np.zeros((h, w), dtype=np.uint8)
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return MaskPrediction(
            mask=empty,
            has_prediction=False,
            num_instances=0,
            num_instances_used=0,
            mask_area_px=0,
            max_conf=None,
            warning="sin_mascaras_detectadas",
        )

    polygons = list(masks.xy)
    class_ids = _result_class_ids(result)
    confs = _result_confs(result)
    candidates: list[tuple[np.ndarray, int, float | None]] = []

    for idx, poly in enumerate(polygons):
        predicted_class = class_ids[idx] if idx < len(class_ids) else class_id
        if predicted_class != class_id:
            continue

        instance_mask = _polygon_to_mask(poly, (h, w))
        area = int(instance_mask.sum())
        if area <= 0:
            continue

        conf = confs[idx] if idx < len(confs) else None
        candidates.append((instance_mask, area, conf))

    if not candidates:
        return MaskPrediction(
            mask=empty,
            has_prediction=False,
            num_instances=len(polygons),
            num_instances_used=0,
            mask_area_px=0,
            max_conf=max(confs) if confs else None,
            warning="sin_instancias_validas_de_carga",
        )

    if mask_mode == "largest":
        selected = [max(candidates, key=lambda item: item[1])]
    else:
        selected = candidates

    merged = np.zeros((h, w), dtype=np.uint8)
    for instance_mask, _, _ in selected:
        merged = np.maximum(merged, instance_mask)

    selected_confs = [conf for _, _, conf in selected if conf is not None]
    return MaskPrediction(
        mask=merged,
        has_prediction=True,
        num_instances=len(candidates),
        num_instances_used=len(selected),
        mask_area_px=int(merged.sum()),
        max_conf=max(selected_confs) if selected_confs else None,
        warning="",
    )


def binary_to_rgb_mask(
    binary_mask: np.ndarray,
    class_rgb: list[int],
    background_rgb: list[int],
) -> np.ndarray:
    rgb = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    rgb[:, :] = np.asarray(background_rgb, dtype=np.uint8)
    rgb[binary_mask > 0] = np.asarray(class_rgb, dtype=np.uint8)
    return rgb


def write_rgb_mask(path: Path, rgb_mask: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe la mascara {path}. Usa --overwrite o elige otra carpeta."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr_mask):
        raise OSError(f"No se pudo escribir la mascara: {path}")


def write_manifest(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe el manifest {path}. Usa --overwrite o elige otra ruta."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "image_path",
        "mask_path",
        "has_prediction",
        "num_instances",
        "num_instances_used",
        "mask_area_px",
        "max_conf",
        "warning",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)

    model_path = resolve_repo_path(args.model)
    images_dir = resolve_repo_path(args.images_dir)
    default_masks_dir = get_nested(
        config,
        ("prediction", "analytic_masks_dir"),
        "model_testing/analytic_masks",
    )
    masks_dir = resolve_repo_path(args.masks_dir or default_masks_dir)
    manifest_path = resolve_repo_path(args.manifest) if args.manifest else masks_dir / "manifest.csv"

    validate_inputs(model_path, images_dir)
    ensure_dir(masks_dir)

    images = list_files(images_dir, IMAGE_EXTENSIONS, recursive=args.recursive)
    if not images:
        raise FileNotFoundError(f"No se encontraron imagenes en: {images_dir}")

    duplicates = find_duplicate_stems(images)
    if duplicates:
        examples = ", ".join(sorted(duplicates.keys())[:5])
        raise ValueError(
            "Hay imagenes con stem duplicado. El analitico empareja imagen/mascara "
            f"por stem, por lo que deben ser unicos. Ejemplos: {examples}"
        )

    output_ext = normalize_output_ext(args.output_ext)
    class_rgb = list(get_nested(config, ("dataset", "class_rgb"), [102, 255, 102]))
    background_rgb = list(get_nested(config, ("dataset", "background_rgb"), [0, 0, 0]))
    class_id = (
        args.class_id
        if args.class_id is not None
        else int(get_nested(config, ("dataset", "class_id"), 0))
    )
    conf = args.conf if args.conf is not None else float(get_nested(config, ("prediction", "conf"), 0.35))
    imgsz = args.imgsz or int(get_nested(config, ("training", "imgsz"), 640))
    device = args.device or get_nested(config, ("training", "device"), "0")
    mask_mode = args.mask_mode or str(
        get_nested(config, ("prediction", "analytic_mask_mode"), "largest")
    )
    if mask_mode not in {"largest", "merge_all"}:
        raise ValueError("prediction.analytic_mask_mode debe ser 'largest' o 'merge_all'.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("No esta instalado ultralytics.") from exc

    model = YOLO(str(model_path))
    manifest_rows: list[dict[str, Any]] = []

    processed = 0
    for batch in chunked(images, args.batch_size):
        results = model.predict(
            source=[str(path) for path in batch],
            task="segment",
            conf=conf,
            imgsz=imgsz,
            device=device,
            save=False,
            verbose=False,
        )

        for result in results:
            image_path = Path(result.path)
            mask_path = masks_dir / f"{image_path.stem}{output_ext}"
            prediction = result_to_binary_mask(result, class_id=class_id, mask_mode=mask_mode)
            rgb_mask = binary_to_rgb_mask(prediction.mask, class_rgb, background_rgb)
            write_rgb_mask(mask_path, rgb_mask, overwrite=args.overwrite)

            manifest_rows.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "has_prediction": prediction.has_prediction,
                    "num_instances": prediction.num_instances,
                    "num_instances_used": prediction.num_instances_used,
                    "mask_area_px": prediction.mask_area_px,
                    "max_conf": "" if prediction.max_conf is None else f"{prediction.max_conf:.6f}",
                    "warning": prediction.warning,
                }
            )
            processed += 1

        print(f"Mascaras generadas {processed}/{len(images)}")

    write_manifest(manifest_path, manifest_rows, overwrite=args.overwrite)
    summary = {
        "images": len(images),
        "masks_written": processed,
        "with_prediction": sum(1 for row in manifest_rows if row["has_prediction"]),
        "without_prediction": sum(1 for row in manifest_rows if not row["has_prediction"]),
        "masks_dir": str(masks_dir),
        "manifest": str(manifest_path),
        "class_rgb": class_rgb,
        "background_rgb": background_rgb,
        "class_id": class_id,
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
        "mask_mode": mask_mode,
    }
    summary_path = masks_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Ya existe el resumen {summary_path}. Usa --overwrite o elige otra carpeta."
        )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Mascaras: {masks_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Resumen: {summary_path}")
    print(f"Con prediccion: {summary['with_prediction']}/{summary['images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
