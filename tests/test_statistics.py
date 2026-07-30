from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cone_load.statistics import (  # noqa: E402
    indicator_distribution_rows,
    percentile,
    summarize_indicators,
)


class StatisticsTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertAlmostEqual(percentile(values, 0.25), 1.75)
        self.assertAlmostEqual(percentile(values, 0.50), 2.50)
        self.assertAlmostEqual(percentile(values, 0.75), 3.25)

    def test_summary_ignores_invalid_rows(self) -> None:
        rows = [
            {
                "is_valid": "True",
                "peak_x_norm": "-0.2",
                "area_symmetry": "-0.4",
                "centroid_x_norm": "-0.1",
            },
            {
                "is_valid": "True",
                "peak_x_norm": "0.2",
                "area_symmetry": "0.4",
                "centroid_x_norm": "0.1",
            },
            {
                "is_valid": "False",
                "peak_x_norm": "0.5",
                "area_symmetry": "1.0",
                "centroid_x_norm": "0.5",
            },
        ]

        summary = summarize_indicators(rows)

        self.assertEqual(summary["peak_x_norm"]["count"], 2)
        self.assertAlmostEqual(float(summary["peak_x_norm"]["median"]), 0.0)
        self.assertAlmostEqual(float(summary["area_symmetry"]["median"]), 0.0)
        self.assertAlmostEqual(float(summary["centroid_x_norm"]["median"]), 0.0)

    def test_histograms_preserve_valid_observation_count(self) -> None:
        rows = [
            {
                "is_valid": True,
                "peak_x_norm": "-0.1",
                "area_symmetry": "-0.2",
                "centroid_x_norm": "-0.05",
            },
            {
                "is_valid": True,
                "peak_x_norm": "0.1",
                "area_symmetry": "0.2",
                "centroid_x_norm": "0.05",
            },
            {
                "is_valid": False,
                "peak_x_norm": "0.3",
                "area_symmetry": "0.6",
                "centroid_x_norm": "0.2",
            },
        ]

        distribution = indicator_distribution_rows(
            rows,
            scope_type="test",
            scope_id="test",
            scope_name="test",
            bins=20,
        )

        self.assertEqual(len(distribution), 60)
        for indicator in ("peak_x_norm", "area_symmetry", "centroid_x_norm"):
            count = sum(
                int(row["count"])
                for row in distribution
                if row["indicator"] == indicator
            )
            self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
