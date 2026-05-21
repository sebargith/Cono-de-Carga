from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import contextlib
import io
import numpy as np
import warnings

from cone_load.io import get_config_value

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(io.StringIO()):
            from scipy.signal import find_peaks as scipy_find_peaks
            from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - fallback for minimal environments
    scipy_find_peaks = None
    savgol_filter = None


@dataclass
class ProfileResult:
    x: np.ndarray
    y_raw: np.ndarray
    y_smooth: np.ndarray
    peak_index: int | None
    peak_indices: np.ndarray


def extract_upper_profile(binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = binary_mask > 0
    columns_with_load = np.any(mask, axis=0)
    x = np.flatnonzero(columns_with_load)
    if x.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    y_top = np.argmax(mask[:, x], axis=0)
    return x.astype(np.int32), y_top.astype(np.float32)


def _make_odd_window(requested: int, n: int) -> int:
    if n < 3:
        return 0
    window = min(max(3, requested), n)
    if window % 2 == 0:
        window -= 1
    return window if window >= 3 else 0


def smooth_profile(y: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float32)

    method = str(get_config_value(config, ("profile", "smoothing", "method"), "savgol"))
    if method == "savgol" and savgol_filter is not None:
        requested = int(
            get_config_value(config, ("profile", "smoothing", "window_length"), 31)
        )
        polyorder = int(get_config_value(config, ("profile", "smoothing", "polyorder"), 2))
        window = _make_odd_window(requested, int(y.size))
        if window > polyorder:
            return savgol_filter(y, window_length=window, polyorder=polyorder).astype(
                np.float32
            )

    requested = int(
        get_config_value(config, ("profile", "smoothing", "moving_average_window"), 15)
    )
    window = _make_odd_window(requested, int(y.size))
    if window == 0:
        return y.astype(np.float32)
    pad = window // 2
    padded = np.pad(y.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _fallback_find_peaks(signal: np.ndarray, min_distance: int) -> np.ndarray:
    if signal.size < 3:
        return np.array([], dtype=np.int32)
    candidates = []
    last = -min_distance
    for i in range(1, signal.size - 1):
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            if i - last >= min_distance:
                candidates.append(i)
                last = i
            elif candidates and signal[i] > signal[candidates[-1]]:
                candidates[-1] = i
                last = i
    return np.asarray(candidates, dtype=np.int32)


def detect_profile_peaks(y_smooth: np.ndarray, config: dict[str, Any]) -> tuple[int | None, np.ndarray]:
    if y_smooth.size == 0:
        return None, np.array([], dtype=np.int32)

    main_idx = int(np.argmin(y_smooth))
    prominence = float(get_config_value(config, ("profile", "peaks", "prominence_px"), 8.0))
    distance = int(get_config_value(config, ("profile", "peaks", "min_distance_px"), 20))
    signal = -y_smooth.astype(np.float32)

    if scipy_find_peaks is not None:
        peak_indices, _ = scipy_find_peaks(
            signal,
            prominence=prominence,
            distance=max(1, distance),
        )
        peak_indices = peak_indices.astype(np.int32)
    else:
        peak_indices = _fallback_find_peaks(signal, max(1, distance))

    if peak_indices.size == 0 or not np.any(np.abs(peak_indices - main_idx) <= max(1, distance // 2)):
        peak_indices = np.unique(np.append(peak_indices, main_idx)).astype(np.int32)

    peak_indices = np.sort(peak_indices)
    return main_idx, peak_indices


def compute_profile_result(binary_mask: np.ndarray, config: dict[str, Any]) -> ProfileResult:
    x, y_raw = extract_upper_profile(binary_mask)
    y_smooth = smooth_profile(y_raw, config)
    peak_index, peak_indices = detect_profile_peaks(y_smooth, config)
    return ProfileResult(
        x=x,
        y_raw=y_raw,
        y_smooth=y_smooth,
        peak_index=peak_index,
        peak_indices=peak_indices,
    )


def profile_roughness(y_raw: np.ndarray, y_smooth: np.ndarray, bbox_height: int) -> float:
    if y_raw.size == 0 or y_raw.size != y_smooth.size or bbox_height <= 0:
        return float("nan")
    residual = y_raw.astype(np.float32) - y_smooth.astype(np.float32)
    return float(np.std(residual) / float(bbox_height))
