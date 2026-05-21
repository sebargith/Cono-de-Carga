from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import yaml


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuracion: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"La configuracion debe ser un diccionario YAML: {config_path}")
    return data


def ensure_dir(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def validate_existing_dir(path: str | Path, name: str) -> Path:
    dir_path = Path(path)
    if not dir_path.exists():
        raise FileNotFoundError(f"No existe {name}: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{name} no es un directorio: {dir_path}")
    return dir_path


def list_files(
    directory: str | Path,
    extensions: tuple[str, ...] | list[str],
    recursive: bool = False,
) -> list[Path]:
    root = Path(directory)
    lower_exts = {ext.lower() for ext in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in lower_exts
    )


def build_stem_index(paths: list[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in paths:
        stem = path.stem
        if stem in index:
            duplicates.setdefault(stem, [index[stem]]).append(path)
            continue
        index[stem] = path
    return index, duplicates


def read_image_bgr(path: str | Path):
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    return image


def read_mask_rgb(path: str | Path):
    mask_path = Path(path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"No se pudo leer la mascara: {mask_path}")
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    if mask.shape[2] == 4:
        return cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
    if mask.shape[2] == 3:
        return cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Formato de mascara no soportado: {mask_path}")


def get_config_value(config: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

