from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import zipfile
from pathlib import Path


SPLITS = ("train", "val", "test")
REQUIRED_DIRS = ("JPEGImages", "SegmentationClass")
OPTIONAL_DIRS = ("SegmentationObject", "ImageSets")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Combina multiples ZIPs de CVAT Segmentation Mask en carpetas train/val/test "
            "listas para usar en perception/data/raw."
        )
    )
    parser.add_argument(
        "--input-root",
        default=str(root / "input"),
        help="Carpeta con subcarpetas train/val/test que contienen ZIPs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(root / "output"),
        help="Carpeta donde se escriben train/val/test combinados.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Borra output-root antes de escribir.")
    return parser.parse_args()


def find_cvat_root(extracted_dir: Path) -> Path:
    candidates = [extracted_dir]
    candidates.extend(path for path in extracted_dir.rglob("*") if path.is_dir())
    for candidate in candidates:
        if all((candidate / required).is_dir() for required in REQUIRED_DIRS):
            return candidate
    raise FileNotFoundError(
        f"No se encontro estructura CVAT con {REQUIRED_DIRS} dentro de {extracted_dir}"
    )


def read_labelmap(cvat_root: Path) -> str:
    path = cvat_root / "labelmap.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def safe_name(zip_path: Path, source_file: Path, cvat_root: Path) -> str:
    relative = source_file.relative_to(cvat_root)
    stem = "__".join(relative.with_suffix("").parts)
    prefix = zip_path.stem.replace(" ", "_")
    return f"{prefix}__{stem}{source_file.suffix.lower()}"


def copy_required_files(
    zip_path: Path,
    cvat_root: Path,
    output_split_dir: Path,
    manifest_rows: list[dict[str, str]],
    split: str,
) -> tuple[int, int]:
    images_out = output_split_dir / "JPEGImages"
    masks_out = output_split_dir / "SegmentationClass"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    image_count = 0
    mask_count = 0
    masks_by_stem = {
        path.stem: path for path in sorted((cvat_root / "SegmentationClass").glob("*")) if path.is_file()
    }

    for image_path in sorted((cvat_root / "JPEGImages").glob("*")):
        if not image_path.is_file():
            continue
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            manifest_rows.append(
                {
                    "split": split,
                    "zip": str(zip_path),
                    "status": "missing_mask",
                    "image": str(image_path),
                    "mask": "",
                    "output_stem": "",
                }
            )
            continue

        image_name = safe_name(zip_path, image_path, cvat_root)
        mask_name = f"{Path(image_name).stem}{mask_path.suffix.lower()}"
        shutil.copy2(image_path, images_out / image_name)
        shutil.copy2(mask_path, masks_out / mask_name)
        image_count += 1
        mask_count += 1
        manifest_rows.append(
            {
                "split": split,
                "zip": str(zip_path),
                "status": "written",
                "image": str(image_path),
                "mask": str(mask_path),
                "output_stem": Path(image_name).stem,
            }
        )

    return image_count, mask_count


def copy_auxiliary_dirs(zip_path: Path, cvat_root: Path, output_split_dir: Path) -> None:
    for dirname in OPTIONAL_DIRS:
        source_dir = cvat_root / dirname
        if not source_dir.exists():
            continue
        target_dir = output_split_dir / dirname / zip_path.stem
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)


def write_labelmap_once(labelmaps: list[str], output_split_dir: Path) -> None:
    clean_labelmaps = [value for value in labelmaps if value]
    if not clean_labelmaps:
        return
    first = clean_labelmaps[0]
    mismatches = [value for value in clean_labelmaps[1:] if value != first]
    if mismatches:
        raise ValueError("Los ZIPs tienen labelmap.txt distintos. Revisa las clases antes de combinar.")
    (output_split_dir / "labelmap.txt").write_text(first + "\n", encoding="utf-8")


def merge_split(input_split_dir: Path, output_split_dir: Path, split: str) -> dict[str, int]:
    zip_paths = sorted(input_split_dir.glob("*.zip"))
    output_split_dir.mkdir(parents=True, exist_ok=True)
    labelmaps: list[str] = []
    manifest_rows: list[dict[str, str]] = []
    total_images = 0
    total_masks = 0

    for zip_path in zip_paths:
        with tempfile.TemporaryDirectory(prefix=f"cvat_{split}_") as tmp:
            tmp_dir = Path(tmp)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
            cvat_root = find_cvat_root(tmp_dir)
            labelmaps.append(read_labelmap(cvat_root))
            images, masks = copy_required_files(
                zip_path,
                cvat_root,
                output_split_dir,
                manifest_rows,
                split,
            )
            copy_auxiliary_dirs(zip_path, cvat_root, output_split_dir)
            total_images += images
            total_masks += masks

    write_labelmap_once(labelmaps, output_split_dir)
    manifest_path = output_split_dir / "merge_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "zip", "status", "image", "mask", "output_stem"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {"zips": len(zip_paths), "images": total_images, "masks": total_masks}


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"No existe input-root: {input_root}")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in SPLITS:
        input_split_dir = input_root / split
        if not input_split_dir.exists():
            input_split_dir.mkdir(parents=True, exist_ok=True)
        output_split_dir = output_root / split
        summary[split] = merge_split(input_split_dir, output_split_dir, split)

    print("Merge CVAT ZIPs completado")
    for split, values in summary.items():
        print(f"{split}: {values['zips']} zips, {values['images']} imagenes, {values['masks']} mascaras")
    print(f"Salida: {output_root}")
    print("Copia train/val/test desde output hacia perception/data/raw o usa output directamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

