from __future__ import annotations

from typing import Any

import numpy as np

from cone_load.io import get_config_value


NO_EVALUABLE = "no_evaluable"


def classify_preliminary(features: dict[str, Any], config: dict[str, Any]) -> str:
    if not bool(features.get("is_valid", False)):
        return NO_EVALUABLE

    peak_x_norm = features.get("peak_x_norm", np.nan)
    area_symmetry = features.get("area_symmetry", np.nan)

    if not np.isfinite(peak_x_norm) or not np.isfinite(area_symmetry):
        return NO_EVALUABLE

    threshold_peak = float(get_config_value(config, ("rules", "threshold_peak"), 0.12))
    threshold_area = float(get_config_value(config, ("rules", "threshold_area"), 0.12))
    peak_vote_weight = float(get_config_value(config, ("rules", "peak_vote_weight"), 0.6))
    area_vote_weight = float(get_config_value(config, ("rules", "area_vote_weight"), 0.4))

    left_score = 0.0
    right_score = 0.0

    if float(peak_x_norm) < -threshold_peak:
        left_score += peak_vote_weight
    elif float(peak_x_norm) > threshold_peak:
        right_score += peak_vote_weight

    if float(area_symmetry) < -threshold_area:
        left_score += area_vote_weight
    elif float(area_symmetry) > threshold_area:
        right_score += area_vote_weight

    if left_score == 0.0 and right_score == 0.0:
        return "centrado"
    if left_score >= right_score:
        return "izquierda"
    return "derecha"
