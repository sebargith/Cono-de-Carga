from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from cone_load.features import analyze_binary_mask
from cone_load.io import load_config
from cone_load.masks import clean_load_mask
from cone_load.rules import classify_preliminary
from demo_synthetic import example_masks, run_demo
from run_pipeline_from_videos import decide_video_verdict


class AnalyticTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'configs' / 'analytic_config.yaml')

    def test_synthetic_shapes(self):
        for expected, mask in example_masks().items():
            with self.subTest(expected=expected):
                mask, warnings = clean_load_mask(mask, self.config)
                row = analyze_binary_mask(expected, mask, self.config, warnings=warnings).features
                self.assertEqual(row['predicted_class'], expected)

    def test_centered_geometry(self):
        row = analyze_binary_mask('center', example_masks()['centrado'], self.config).features
        self.assertAlmostEqual(row['area_symmetry'], 0, places=2)
        self.assertAlmostEqual(row['centroid_x_norm'], 0, places=2)
        self.assertLess(abs(row['peak_x_norm']), 0.02)

    def test_empty_and_small_masks(self):
        for size, reason in [(0, 'mascara_vacia'), (3, 'mascara_muy_pequena')]:
            with self.subTest(size=size):
                mask = np.zeros((100, 100), dtype=np.uint8)
                mask[20:20+size, 20:20+size] = 1
                row = analyze_binary_mask('invalid', mask, self.config).features
                self.assertEqual(row['predicted_class'], 'no_evaluable')
                self.assertEqual(row['error_reason'], reason)

    def test_border_is_rejected(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:50, 20:80] = 1
        row = analyze_binary_mask('border', mask, self.config).features
        self.assertEqual(row['error_reason'], 'mascara_toca_borde_imagen')
        self.assertFalse(row['is_valid'])

    def test_cleanup_removes_isolated_noise(self):
        mask = example_masks()['centrado']
        mask[10, 10] = 1
        cleaned, _ = clean_load_mask(mask, self.config)
        self.assertEqual(cleaned[10, 10], 0)
        self.assertGreater(cleaned.sum(), 100)

    def classify(self, peak, area, valid=True):
        return classify_preliminary(dict(is_valid=valid, peak_x_norm=peak, area_symmetry=area), self.config)

    def test_rule_thresholds_are_strict(self):
        self.assertEqual(self.classify(0.06, 0.07), 'centrado')
        self.assertEqual(self.classify(-0.06, -0.07), 'centrado')
        self.assertEqual(self.classify(0.061, 0), 'derecha')
        self.assertEqual(self.classify(-0.061, 0), 'izquierda')

    def test_nonfinite_or_invalid_is_rejected(self):
        self.assertEqual(self.classify(float('nan'), 0), 'no_evaluable')
        self.assertEqual(self.classify(0, float('inf')), 'no_evaluable')
        self.assertEqual(self.classify(0, 0, False), 'no_evaluable')

    def test_existing_tie_break_favors_left(self):
        # Characterization of existing behavior, not a desired new policy.
        self.assertEqual(self.classify(-0.2, 0.2), 'izquierda')
        self.assertEqual(self.classify(0.2, -0.2), 'izquierda')

    def test_demo_outputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'demo'
            rows = run_demo(output)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r['predicted'] == r['expected'] for r in rows))
            self.assertEqual(len(list(output.glob('*.png'))), 4)
            self.assertTrue((output / 'results.csv').is_file())
            with self.assertRaises(ValueError):
                run_demo(output)


class VideoTests(unittest.TestCase):
    def verdict(self, **changes):
        args = dict(valid_ratio=1, centered_ratio=0.4, left_ratio=0.3, right_ratio=0.3,
                    median_peak_x_norm=0, median_area_symmetry=0,
                    min_valid_ratio=0.5, majority_ratio=0.6, threshold_peak=0.06, threshold_area=0.07)
        args.update(changes)
        return decide_video_verdict(**args)

    def test_invalid_video(self):
        self.assertEqual(self.verdict(valid_ratio=0.49)[0], 'no_evaluable')

    def test_majority_at_boundary(self):
        result = self.verdict(valid_ratio=0.5, centered_ratio=0.2, left_ratio=0.6, right_ratio=0.2)
        self.assertEqual(result[0], 'tendencia_izquierda')
        self.assertEqual(result[1], 0.6)

    def test_contradictory_medians_preserve_existing_left_priority(self):
        result = self.verdict(median_peak_x_norm=0.2, median_area_symmetry=-0.2)
        self.assertEqual(result[0], 'tendencia_izquierda')

    def test_no_majority_and_missing_medians(self):
        self.assertEqual(self.verdict(median_peak_x_norm=float('nan'))[2], 'sin_mayoria_clara')


if __name__ == '__main__':
    unittest.main()
