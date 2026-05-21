from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from cone_load.io import get_config_value
from cone_load.profile import ProfileResult, compute_profile_result, profile_roughness
from cone_load.rules import NO_EVALUABLE, classify_preliminary


FEATURE_COLUMNS = [
    "image_name",
    "mask_area_px",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "bbox_width",
    "bbox_height",
    "bbox_aspect_ratio",
    "centroid_x",
    "centroid_y",
    "centroid_x_norm",
    "peak_x",
    "peak_y",
    "peak_x_norm",
    "area_left",
    "area_right",
    "area_symmetry",
    "profile_roughness",
    "num_peaks",
    "predicted_class",
    "is_valid",
]

EXTRA_COLUMNS = [
    "error_reason",
    "warnings",
    "mask_path",
    "overlay_path",
    "unexpected_pixel_count",
    "unexpected_color_count",
]


@dataclass
class AnalyticResult:
    features: dict[str, Any]
    profile: ProfileResult | None = None
    warnings: list[str] = field(default_factory=list)


def empty_feature_row(
    image_name: str,
    reason: str,
    mask_path: str = "",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {column: np.nan for column in FEATURE_COLUMNS + EXTRA_COLUMNS}
    row["image_name"] = image_name
    row["predicted_class"] = NO_EVALUABLE
    row["is_valid"] = False
    row["error_reason"] = reason
    row["warnings"] = ";".join(warnings or [])
    row["mask_path"] = mask_path
    row["overlay_path"] = ""
    row["unexpected_pixel_count"] = 0
    row["unexpected_color_count"] = 0
    return row


def _bbox_from_mask(binary_mask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    ys, xs = np.where(binary_mask > 0)
    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())
    width = int(xmax - xmin + 1)
    height = int(ymax - ymin + 1)
    return xmin, ymin, xmax, ymax, width, height


def _centroid_from_mask(binary_mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(binary_mask.astype(np.uint8), binaryImage=True)
    if moments["m00"] == 0:
        return float("nan"), float("nan")
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def _area_split(binary_mask: np.ndarray, bbox_center_x: float) -> tuple[int, int, float]:
    ys, xs = np.where(binary_mask > 0)
    left = int(np.sum(xs < bbox_center_x))
    right = int(np.sum(xs > bbox_center_x))
    denom = left + right
    symmetry = float((right - left) / denom) if denom else float("nan")
    return left, right, symmetry


def _normalized_x(x: float, bbox_center_x: float, bbox_width: int) -> float:
    if bbox_width <= 0 or not np.isfinite(x):
        return float("nan")
    return float((x - bbox_center_x) / float(bbox_width))


def analyze_binary_mask(
    image_name: str,
    binary_mask: np.ndarray,
    config: dict[str, Any],
    mask_path: str = "",
    warnings: list[str] | None = None,
    unexpected_pixel_count: int = 0,
    unexpected_color_count: int = 0,
) -> AnalyticResult:
    row = empty_feature_row(image_name, "", mask_path=mask_path, warnings=warnings)
    row["unexpected_pixel_count"] = unexpected_pixel_count
    row["unexpected_color_count"] = unexpected_color_count

    mask = (binary_mask > 0).astype(np.uint8)
    area = int(mask.sum())
    row["mask_area_px"] = area

    min_area = int(get_config_value(config, ("geometry", "min_area_px"), 100))
    if area == 0:
        row["error_reason"] = "mascara_vacia"
        return AnalyticResult(row, warnings=warnings or [])
    if area < min_area:
        row["error_reason"] = "mascara_muy_pequena"
        return AnalyticResult(row, warnings=warnings or [])

    xmin, ymin, xmax, ymax, width, height = _bbox_from_mask(mask)
    row.update(
        {
            "bbox_xmin": xmin,
            "bbox_ymin": ymin,
            "bbox_xmax": xmax,
            "bbox_ymax": ymax,
            "bbox_width": width,
            "bbox_height": height,
            "bbox_aspect_ratio": float(width / height) if height else float("nan"),
        }
    )

    min_width = int(get_config_value(config, ("geometry", "min_bbox_width"), 10))
    min_height = int(get_config_value(config, ("geometry", "min_bbox_height"), 10))
    if width < min_width or height < min_height:
        row["error_reason"] = "bbox_demasiado_pequeno"
        return AnalyticResult(row, warnings=warnings or [])

    bbox_center_x = xmin + (width - 1) / 2.0
    centroid_x, centroid_y = _centroid_from_mask(mask)
    area_left, area_right, area_symmetry = _area_split(mask, bbox_center_x)

    row.update(
        {
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "centroid_x_norm": _normalized_x(centroid_x, bbox_center_x, width),
            "area_left": area_left,
            "area_right": area_right,
            "area_symmetry": area_symmetry,
        }
    )

    profile = compute_profile_result(mask, config)
    min_profile_points = int(get_config_value(config, ("geometry", "min_profile_points"), 20))
    if profile.x.size < min_profile_points or profile.peak_index is None:
        row["error_reason"] = "perfil_superior_insuficiente"
        return AnalyticResult(row, profile=profile, warnings=warnings or [])

    peak_idx = int(profile.peak_index)
    peak_x = float(profile.x[peak_idx])
    peak_y = float(profile.y_smooth[peak_idx])
    roughness = profile_roughness(profile.y_raw, profile.y_smooth, height)

    row.update(
        {
            "peak_x": peak_x,
            "peak_y": peak_y,
            "peak_x_norm": _normalized_x(peak_x, bbox_center_x, width),
            "profile_roughness": roughness,
            "num_peaks": int(profile.peak_indices.size),
            "is_valid": True,
            "error_reason": "",
        }
    )
    row["predicted_class"] = classify_preliminary(row, config)
    row["warnings"] = ";".join(warnings or [])
    return AnalyticResult(row, profile=profile, warnings=warnings or [])

