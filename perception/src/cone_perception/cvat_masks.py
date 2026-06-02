from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cone_perception.io import get_nested


@dataclass
class MaskToYoloResult:
    lines: list[str]
    binary_area_px: int
    contour_count: int
    kept_contour_count: int
    warnings: list[str]


def rgb_to_binary(mask_rgb: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    target = np.asarray(get_nested(config, ("dataset", "class_rgb"), [102, 255, 102]), dtype=np.int16)
    background = np.asarray(
        get_nested(config, ("dataset", "background_rgb"), [0, 0, 0]), dtype=np.int16
    )
    tolerance = int(get_nested(config, ("dataset", "color_tolerance"), 0))

    mask_i16 = mask_rgb.astype(np.int16)
    is_target = np.all(np.abs(mask_i16 - target) <= tolerance, axis=2)
    is_background = np.all(np.abs(mask_i16 - background) <= tolerance, axis=2)
    unexpected = ~(is_target | is_background)

    warnings: list[str] = []
    if int(unexpected.sum()) > 0:
        warnings.append(f"colores_inesperados_px={int(unexpected.sum())}")

    return is_target.astype(np.uint8), warnings


def clean_binary_mask(binary_mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    mask = (binary_mask > 0).astype(np.uint8)
    open_kernel = int(get_nested(config, ("dataset", "morph_open_kernel"), 0))
    close_kernel = int(get_nested(config, ("dataset", "morph_close_kernel"), 0))
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(np.uint8)


def _downsample_points(points: np.ndarray, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int32)
    return points[idx]


def contour_to_yolo_polygon(
    contour: np.ndarray,
    image_width: int,
    image_height: int,
    epsilon_fraction: float,
    max_points: int | None,
) -> list[float] | None:
    if contour.shape[0] < 3:
        return None

    arc = cv2.arcLength(contour, closed=True)
    epsilon = max(0.0, float(epsilon_fraction)) * arc
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    points = approx.reshape(-1, 2)
    points = _downsample_points(points, max_points)

    if points.shape[0] < 3:
        return None

    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    flat: list[float] = []
    for x, y in points:
        flat.append(float(np.clip(x / width, 0.0, 1.0)))
        flat.append(float(np.clip(y / height, 0.0, 1.0)))
    return flat


def mask_to_yolo_lines(
    binary_mask: np.ndarray,
    class_id: int,
    image_width: int,
    image_height: int,
    config: dict[str, Any],
    inherited_warnings: list[str] | None = None,
) -> MaskToYoloResult:
    warnings = list(inherited_warnings or [])
    mask = clean_binary_mask(binary_mask, config)
    area = int(mask.sum())

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    min_area = float(get_nested(config, ("dataset", "min_contour_area_px"), 50))
    keep_largest = bool(get_nested(config, ("dataset", "keep_largest_component"), True))
    epsilon_fraction = float(get_nested(config, ("dataset", "polygon_epsilon_fraction"), 0.002))
    max_points = get_nested(config, ("dataset", "max_polygon_points"), 200)
    max_points = None if max_points is None else int(max_points)

    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if keep_largest and valid_contours:
        valid_contours = [max(valid_contours, key=cv2.contourArea)]

    lines: list[str] = []
    for contour in valid_contours:
        polygon = contour_to_yolo_polygon(
            contour,
            image_width=image_width,
            image_height=image_height,
            epsilon_fraction=epsilon_fraction,
            max_points=max_points,
        )
        if polygon is None:
            warnings.append("contorno_descartado_por_menos_de_3_puntos")
            continue
        coords = " ".join(f"{value:.6f}" for value in polygon)
        lines.append(f"{class_id} {coords}")

    if area == 0:
        warnings.append("mascara_vacia")
    elif not lines:
        warnings.append("sin_contornos_yolo_validos")

    return MaskToYoloResult(
        lines=lines,
        binary_area_px=area,
        contour_count=contour_count,
        kept_contour_count=len(lines),
        warnings=warnings,
    )

