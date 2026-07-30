from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cone_load.features import AnalyticResult
from cone_load.io import ensure_dir, get_config_value


def _write_image(path: Path, image: np.ndarray) -> bool:
    extension = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def _finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _fmt(value: Any, digits: int = 3) -> str:
    if not _finite_number(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _draw_text_block(image: np.ndarray, lines: list[str], x: int = 12, y: int = 24) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    line_height = 22
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max((size[0] for size in sizes), default=0) + 18
    height = line_height * len(lines) + 12

    overlay = image.copy()
    cv2.rectangle(overlay, (x - 8, y - 20), (x - 8 + width, y - 20 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, dst=image)

    for idx, line in enumerate(lines):
        yy = y + idx * line_height
        cv2.putText(image, line, (x, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_bbox_and_guides(canvas: np.ndarray, features: dict[str, Any]) -> None:
    required = ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "bbox_width"]
    if not all(_finite_number(features.get(key)) for key in required):
        return

    xmin = int(round(float(features["bbox_xmin"])))
    ymin = int(round(float(features["bbox_ymin"])))
    xmax = int(round(float(features["bbox_xmax"])))
    ymax = int(round(float(features["bbox_ymax"])))
    width = int(round(float(features["bbox_width"])))
    center_x = int(round(xmin + (width - 1) / 2.0))

    cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    cv2.line(canvas, (center_x, ymin), (center_x, ymax), (255, 160, 0), 2)


def _draw_centroid_and_peaks(canvas: np.ndarray, result: AnalyticResult) -> None:
    features = result.features
    if _finite_number(features.get("centroid_x")) and _finite_number(features.get("centroid_y")):
        centroid = (
            int(round(float(features["centroid_x"]))),
            int(round(float(features["centroid_y"]))),
        )
        cv2.circle(canvas, centroid, 6, (255, 0, 255), -1)
        cv2.circle(canvas, centroid, 9, (255, 255, 255), 1)

    profile = result.profile
    if profile is None or profile.x.size == 0:
        return

    if profile.x.size > 1:
        smooth_points = np.column_stack([profile.x, profile.y_smooth]).round().astype(np.int32)
        cv2.polylines(canvas, [smooth_points], False, (0, 0, 255), 2, cv2.LINE_AA)

    for idx in profile.peak_indices:
        if idx < 0 or idx >= profile.x.size:
            continue
        point = (int(round(float(profile.x[idx]))), int(round(float(profile.y_smooth[idx]))))
        cv2.circle(canvas, point, 5, (255, 255, 255), -1)
        cv2.circle(canvas, point, 7, (0, 0, 255), 2)

    if profile.peak_index is not None and 0 <= profile.peak_index < profile.x.size:
        idx = int(profile.peak_index)
        point = (int(round(float(profile.x[idx]))), int(round(float(profile.y_smooth[idx]))))
        cv2.drawMarker(
            canvas,
            point,
            (0, 255, 0),
            markerType=cv2.MARKER_DIAMOND,
            markerSize=20,
            thickness=2,
            line_type=cv2.LINE_AA,
        )


def render_overlay(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray | None,
    result: AnalyticResult,
    output_path: str | Path,
    config: dict[str, Any],
) -> Path:
    canvas = image_bgr.copy()
    alpha = float(get_config_value(config, ("visualization", "mask_alpha"), 0.35))

    if binary_mask is not None and binary_mask.shape[:2] == canvas.shape[:2]:
        color_layer = np.zeros_like(canvas)
        color_layer[binary_mask > 0] = (80, 255, 80)
        canvas = cv2.addWeighted(color_layer, alpha, canvas, 1.0 - alpha, 0)

    _draw_bbox_and_guides(canvas, result.features)
    _draw_centroid_and_peaks(canvas, result)

    features = result.features
    lines = [
        str(features.get("image_name", "")),
        f"clase: {features.get('predicted_class', 'no_evaluable')}",
        f"peak_x_norm: {_fmt(features.get('peak_x_norm'))}",
        f"area_symmetry: {_fmt(features.get('area_symmetry'))}",
        f"profile_roughness: {_fmt(features.get('profile_roughness'))}",
        f"num_peaks: {features.get('num_peaks', 'nan')}",
    ]
    if not bool(features.get("is_valid", False)):
        lines.append(f"error: {features.get('error_reason', '')}")

    _draw_text_block(canvas, lines)

    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    ok = _write_image(out_path, canvas)
    if not ok:
        raise ValueError(f"No se pudo guardar overlay: {out_path}")
    return out_path
