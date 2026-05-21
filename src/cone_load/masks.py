from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cone_load.io import get_config_value


@dataclass
class MaskConversion:
    binary: np.ndarray
    warnings: list[str]
    unexpected_pixel_count: int
    unexpected_color_count: int


def rgb_mask_to_binary(mask_rgb: np.ndarray, config: dict[str, Any]) -> MaskConversion:
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError("La mascara debe tener 3 canales RGB.")

    target = np.asarray(
        get_config_value(config, ("mask", "target_rgb"), [102, 255, 102]),
        dtype=np.int16,
    )
    background = np.asarray(
        get_config_value(config, ("mask", "background_rgb"), [0, 0, 0]),
        dtype=np.int16,
    )
    tolerance = int(get_config_value(config, ("mask", "color_tolerance"), 0))
    warn_fraction = float(
        get_config_value(config, ("mask", "warn_unexpected_color_fraction"), 0.0)
    )

    mask_i16 = mask_rgb.astype(np.int16)
    is_target = np.all(np.abs(mask_i16 - target) <= tolerance, axis=2)
    is_background = np.all(np.abs(mask_i16 - background) <= tolerance, axis=2)
    unexpected = ~(is_target | is_background)

    unexpected_pixel_count = int(unexpected.sum())
    unexpected_color_count = 0
    warnings: list[str] = []
    if unexpected_pixel_count:
        colors = np.unique(mask_rgb[unexpected].reshape(-1, 3), axis=0)
        unexpected_color_count = int(colors.shape[0])
        fraction = unexpected_pixel_count / float(mask_rgb.shape[0] * mask_rgb.shape[1])
        if fraction > warn_fraction:
            warnings.append(
                "mascara_con_colores_inesperados:"
                f"{unexpected_pixel_count}_px,{unexpected_color_count}_colores"
            )

    return MaskConversion(
        binary=is_target.astype(np.uint8),
        warnings=warnings,
        unexpected_pixel_count=unexpected_pixel_count,
        unexpected_color_count=unexpected_color_count,
    )


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    mask = (binary_mask > 0).astype(np.uint8)
    if mask.size == 0:
        return mask

    padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    flood = padded.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood[1:-1, 1:-1] == 0).astype(np.uint8)
    return np.maximum(mask, holes).astype(np.uint8)


def apply_morphology(binary_mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    mask = (binary_mask > 0).astype(np.uint8)
    open_kernel = int(get_config_value(config, ("mask", "morph_open_kernel"), 0))
    close_kernel = int(get_config_value(config, ("mask", "morph_close_kernel"), 0))

    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if bool(get_config_value(config, ("mask", "fill_holes"), True)):
        mask = fill_holes(mask)

    return mask.astype(np.uint8)


def largest_connected_component(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = (binary_mask > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return mask, 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask), 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest_area = int(areas[largest_label - 1])
    return (labels == largest_label).astype(np.uint8), largest_area


def clean_load_mask(binary_mask: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    mask = apply_morphology(binary_mask, config)

    if bool(get_config_value(config, ("mask", "keep_largest_component"), True)):
        before_area = int(mask.sum())
        mask, largest_area = largest_connected_component(mask)
        if before_area and largest_area < before_area:
            removed = before_area - largest_area
            warnings.append(f"componentes_secundarios_removidos:{removed}_px")

    return mask.astype(np.uint8), warnings
