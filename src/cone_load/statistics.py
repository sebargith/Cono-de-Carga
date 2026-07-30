from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class IndicatorDefinition:
    name: str
    label: str
    minimum: float
    maximum: float


INDICATORS = (
    IndicatorDefinition(
        name="peak_x_norm",
        label="Posicion relativa del pico",
        minimum=-0.5,
        maximum=0.5,
    ),
    IndicatorDefinition(
        name="area_symmetry",
        label="Simetria interna de area",
        minimum=-1.0,
        maximum=1.0,
    ),
    IndicatorDefinition(
        name="centroid_x_norm",
        label="Posicion relativa del centroide",
        minimum=-0.5,
        maximum=0.5,
    ),
)


def is_true(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "si", "yes"}


def finite_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability debe quedar entre 0 y 1")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]

    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def values_for_indicator(
    rows: Iterable[Mapping[str, Any]],
    indicator: str,
    *,
    valid_only: bool = True,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if valid_only and not is_true(row.get("is_valid")):
            continue
        value = finite_float(row.get(indicator))
        if value is not None:
            values.append(value)
    return values


def describe_values(values: Sequence[float]) -> dict[str, float | int]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return {
            "count": 0,
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
        }
    return {
        "count": len(finite_values),
        "p25": percentile(finite_values, 0.25),
        "median": percentile(finite_values, 0.50),
        "p75": percentile(finite_values, 0.75),
    }


def summarize_indicators(
    rows: Iterable[Mapping[str, Any]],
    *,
    valid_only: bool = True,
) -> dict[str, dict[str, float | int | str]]:
    materialized = list(rows)
    summaries: dict[str, dict[str, float | int | str]] = {}
    for definition in INDICATORS:
        values = values_for_indicator(
            materialized,
            definition.name,
            valid_only=valid_only,
        )
        summaries[definition.name] = {
            "label": definition.label,
            **describe_values(values),
        }
    return summaries


def indicator_summary_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    scope_type: str,
    scope_id: str,
    scope_name: str,
    valid_only: bool = True,
) -> list[dict[str, Any]]:
    summaries = summarize_indicators(rows, valid_only=valid_only)
    output: list[dict[str, Any]] = []
    for definition in INDICATORS:
        summary = summaries[definition.name]
        output.append(
            {
                "scope_type": scope_type,
                "scope_id": scope_id,
                "scope_name": scope_name,
                "indicator": definition.name,
                "indicator_label": definition.label,
                "count": summary["count"],
                "p25": summary["p25"],
                "median": summary["median"],
                "p75": summary["p75"],
            }
        )
    return output


def indicator_distribution_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    scope_type: str,
    scope_id: str,
    scope_name: str,
    bins: int = 20,
    valid_only: bool = True,
) -> list[dict[str, Any]]:
    if bins <= 0:
        raise ValueError("bins debe ser mayor que cero")

    materialized = list(rows)
    output: list[dict[str, Any]] = []
    for definition in INDICATORS:
        values = values_for_indicator(
            materialized,
            definition.name,
            valid_only=valid_only,
        )
        width = (definition.maximum - definition.minimum) / bins
        counts = [0] * bins
        for value in values:
            clipped = min(max(value, definition.minimum), definition.maximum)
            index = int((clipped - definition.minimum) / width)
            index = min(index, bins - 1)
            counts[index] += 1

        for index, count in enumerate(counts):
            bin_start = definition.minimum + index * width
            bin_end = bin_start + width
            output.append(
                {
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                    "scope_name": scope_name,
                    "indicator": definition.name,
                    "indicator_label": definition.label,
                    "bin_index": index,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "count": count,
                }
            )
    return output
