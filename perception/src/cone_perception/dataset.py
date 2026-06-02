from __future__ import annotations

import csv
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cone_perception.cvat_masks import mask_to_yolo_lines, rgb_to_binary
from cone_perception.io import (
    IMAGE_EXTENSIONS,
    MASK_EXTENSIONS,
    build_stem_index,
    ensure_dir,
    get_nested,
    list_files,
    read_mask_rgb,
    validate_dir,
    write_yaml,
)


@dataclass(frozen=True)
class Sample:
    source_name: str
    image_path: Path
    mask_path: Path
    output_stem: str


def sanitize_stem(value: str) -> str:
    value = value.replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def discover_cvat_samples(
    cvat_dir: str | Path,
    recursive: bool = False,
    source_name: str | None = None,
) -> tuple[list[Sample], list[str]]:
    root = validate_dir(cvat_dir, "cvat-dir")
    images_dir = validate_dir(root / "JPEGImages", "JPEGImages")
    masks_dir = validate_dir(root / "SegmentationClass", "SegmentationClass")
    source = sanitize_stem(source_name or root.name)

    images = list_files(images_dir, IMAGE_EXTENSIONS, recursive=recursive)
    masks = list_files(masks_dir, MASK_EXTENSIONS, recursive=recursive)
    mask_index, duplicates = build_stem_index(masks)

    warnings = []
    if duplicates:
        warnings.append(f"{root}: mascaras_duplicadas={len(duplicates)}")

    samples: list[Sample] = []
    used_stems: set[str] = set()
    for image in images:
        mask = mask_index.get(image.stem)
        if mask is None:
            warnings.append(f"sin_mascara:{image.name}")
            continue

        rel_stem = "__".join(image.relative_to(images_dir).with_suffix("").parts)
        output_stem = sanitize_stem(f"{source}__{rel_stem}")
        if output_stem in used_stems:
            output_stem = sanitize_stem(f"{output_stem}__{len(used_stems)}")
        used_stems.add(output_stem)

        samples.append(
            Sample(
                source_name=source,
                image_path=image,
                mask_path=mask,
                output_stem=output_stem,
            )
        )
    return samples, warnings


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    if not samples:
        return {"train": [], "val": [], "test": []}

    ratios_sum = train_ratio + val_ratio + test_ratio
    if ratios_sum <= 0:
        raise ValueError("La suma de train_ratio, val_ratio y test_ratio debe ser mayor que 0.")

    train_ratio /= ratios_sum
    val_ratio /= ratios_sum
    test_ratio /= ratios_sum

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)

    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    if n >= 2 and n_val == 0 and val_ratio > 0:
        n_val = 1
    if n >= 3 and n_test == 0 and test_ratio > 0:
        n_test = 1

    n_train = n - n_val - n_test
    while n_train <= 0 and (n_val > 0 or n_test > 0):
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        n_train = n - n_val - n_test

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def write_dataset_yaml(output_dir: Path, class_names: list[str]) -> Path:
    data = {
        "path": output_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    path = output_dir / "dataset.yaml"
    write_yaml(path, data)
    return path


def convert_samples_to_yolo(
    splits: dict[str, list[Sample]],
    output_dir: str | Path,
    config: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    ensure_dir(out)

    class_names = list(get_nested(config, ("dataset", "names"), ["Carga"]))
    class_id = int(get_nested(config, ("dataset", "class_id"), 0))
    include_empty = bool(get_nested(config, ("dataset", "include_empty_labels"), False))

    manifest_rows: list[dict[str, Any]] = []
    totals = {
        "samples": 0,
        "written": 0,
        "skipped_empty": 0,
        "warnings": 0,
        "by_split": {},
    }

    for split, samples in splits.items():
        images_out = ensure_dir(out / "images" / split)
        labels_out = ensure_dir(out / "labels" / split)
        totals["by_split"][split] = {"input": len(samples), "written": 0}

        for sample in samples:
            totals["samples"] += 1
            warnings: list[str] = []
            try:
                mask_rgb = read_mask_rgb(sample.mask_path)
                h, w = mask_rgb.shape[:2]
                binary, color_warnings = rgb_to_binary(mask_rgb, config)
                result = mask_to_yolo_lines(binary, class_id, w, h, config, color_warnings)
                warnings.extend(result.warnings)

                if not result.lines and not include_empty:
                    totals["skipped_empty"] += 1
                    status = "skipped_empty"
                else:
                    out_image_name = f"{sample.output_stem}{sample.image_path.suffix.lower()}"
                    out_label_name = f"{sample.output_stem}.txt"
                    shutil.copy2(sample.image_path, images_out / out_image_name)
                    with (labels_out / out_label_name).open("w", encoding="utf-8") as f:
                        f.write("\n".join(result.lines))
                        if result.lines:
                            f.write("\n")
                    totals["written"] += 1
                    totals["by_split"][split]["written"] += 1
                    status = "written"

                totals["warnings"] += int(bool(warnings))
                manifest_rows.append(
                    {
                        "split": split,
                        "status": status,
                        "source_name": sample.source_name,
                        "image_path": str(sample.image_path),
                        "mask_path": str(sample.mask_path),
                        "output_stem": sample.output_stem,
                        "mask_area_px": result.binary_area_px,
                        "contour_count": result.contour_count,
                        "kept_contour_count": result.kept_contour_count,
                        "warnings": ";".join(warnings),
                    }
                )
            except Exception as exc:
                totals["warnings"] += 1
                manifest_rows.append(
                    {
                        "split": split,
                        "status": "error",
                        "source_name": sample.source_name,
                        "image_path": str(sample.image_path),
                        "mask_path": str(sample.mask_path),
                        "output_stem": sample.output_stem,
                        "mask_area_px": "",
                        "contour_count": "",
                        "kept_contour_count": "",
                        "warnings": f"{type(exc).__name__}:{exc}",
                    }
                )

    dataset_yaml = write_dataset_yaml(out, class_names)
    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "status",
            "source_name",
            "image_path",
            "mask_path",
            "output_stem",
            "mask_area_px",
            "contour_count",
            "kept_contour_count",
            "warnings",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    totals["dataset_yaml"] = str(dataset_yaml)
    totals["manifest_csv"] = str(manifest_path)
    return totals

