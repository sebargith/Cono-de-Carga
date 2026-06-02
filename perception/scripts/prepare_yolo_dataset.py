from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cone_perception.dataset import convert_samples_to_yolo, discover_cvat_samples, split_samples
from cone_perception.io import get_nested, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara un dataset YOLO de segmentacion desde exportaciones CVAT "
            "Segmentation Mask con JPEGImages/ y SegmentationClass/."
        )
    )
    parser.add_argument(
        "--cvat-dir",
        action="append",
        default=None,
        help="Carpeta de exportacion CVAT. Se puede repetir para combinar datasets.",
    )
    parser.add_argument(
        "--raw-root",
        default=str(ROOT / "data" / "raw"),
        help=(
            "Raiz con carpetas train/val/test. Por defecto: perception/data/raw. "
            "Si existen estas carpetas, se respetan los splits y no se hace split aleatorio."
        ),
    )
    parser.add_argument("--train-cvat-dir", action="append", default=None)
    parser.add_argument("--val-cvat-dir", action="append", default=None)
    parser.add_argument("--test-cvat-dir", action="append", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Carpeta de salida YOLO. Si se omite, usa dataset.output_dir del config.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "perception_config.yaml"),
        help="Archivo de configuracion YAML.",
    )
    parser.add_argument("--recursive", action="store_true", help="Busca archivos en subcarpetas.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para split aleatorio.")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Borra y regenera output-dir si ya existe.",
    )
    return parser.parse_args()


def discover_many(paths: list[str], split_name: str, recursive: bool):
    samples = []
    warnings = []
    for path in paths:
        found, found_warnings = discover_cvat_samples(
            path,
            recursive=recursive,
            source_name=f"{split_name}_{Path(path).name}",
        )
        samples.extend(found)
        warnings.extend(found_warnings)
    return samples, warnings


def default_split_dirs(raw_root: str | Path) -> dict[str, list[str]]:
    root = Path(raw_root)
    split_dirs = {
        "train": root / "train",
        "val": root / "val",
        "test": root / "test",
    }
    return {
        split: [str(path)]
        for split, path in split_dirs.items()
        if path.exists() and (path / "JPEGImages").exists() and (path / "SegmentationClass").exists()
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)

    warnings = []
    output_dir = args.output_dir or get_nested(
        config, ("dataset", "output_dir"), str(ROOT / "data" / "yolo" / "carga_seg")
    )

    explicit_splits = {
        "train": args.train_cvat_dir or [],
        "val": args.val_cvat_dir or [],
        "test": args.test_cvat_dir or [],
    }
    has_explicit_splits = any(explicit_splits.values())

    if has_explicit_splits:
        split_paths = explicit_splits
        split_mode = "splits_explicitos"
    else:
        detected = default_split_dirs(args.raw_root)
        if detected:
            split_paths = {
                "train": detected.get("train", []),
                "val": detected.get("val", []),
                "test": detected.get("test", []),
            }
            split_mode = "raw_root_train_val_test"
        elif args.cvat_dir:
            samples = []
            for cvat_dir in args.cvat_dir:
                found, found_warnings = discover_cvat_samples(cvat_dir, recursive=args.recursive)
                samples.extend(found)
                warnings.extend(found_warnings)

            if not samples:
                raise RuntimeError("No se encontraron pares imagen/mascara para convertir.")

            train_ratio = args.train_ratio
            if train_ratio is None:
                train_ratio = float(get_nested(config, ("split", "train"), 0.7))
            val_ratio = args.val_ratio
            if val_ratio is None:
                val_ratio = float(get_nested(config, ("split", "val"), 0.2))
            test_ratio = args.test_ratio
            if test_ratio is None:
                test_ratio = float(get_nested(config, ("split", "test"), 0.1))
            seed = args.seed
            if seed is None:
                seed = int(get_nested(config, ("split", "seed"), 42))

            splits = split_samples(samples, train_ratio, val_ratio, test_ratio, seed)
            split_mode = "split_aleatorio"
        else:
            raise RuntimeError(
                "No se encontro data. Usa data/raw/train, data/raw/val y opcional data/raw/test, "
                "o entrega --cvat-dir, o usa --train-cvat-dir/--val-cvat-dir/--test-cvat-dir."
            )

    if has_explicit_splits or "split_paths" in locals():
        splits = {}
        for split_name in ("train", "val", "test"):
            samples, found_warnings = discover_many(
                split_paths.get(split_name, []),
                split_name=split_name,
                recursive=args.recursive,
            )
            splits[split_name] = samples
            warnings.extend(found_warnings)

    total_samples = sum(len(samples) for samples in splits.values())
    if total_samples == 0:
        raise RuntimeError("No se encontraron pares imagen/mascara para convertir.")

    summary = convert_samples_to_yolo(splits, output_dir, config, overwrite=args.overwrite)

    print("Preparacion YOLO completada")
    print(f"Modo de split: {split_mode}")
    print(f"Samples encontrados: {total_samples}")
    print(f"Samples escritos: {summary['written']}")
    print(f"Samples vacios omitidos: {summary['skipped_empty']}")
    print(f"Dataset YAML: {summary['dataset_yaml']}")
    print(f"Manifest: {summary['manifest_csv']}")
    print("Split:")
    for split, values in summary["by_split"].items():
        print(f"  {split}: {values['written']}/{values['input']}")
    if warnings:
        print("Advertencias de descubrimiento:")
        for warning in warnings:
            print(f"  - {warning}")
    print("Resumen JSON:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
