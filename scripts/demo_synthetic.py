"""Exercise the real geometric analytic with synthetic masks, without a model.

These shapes are software examples, NOT mining data or evaluation evidence.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cone_load.features import analyze_binary_mask
from cone_load.io import load_config
from cone_load.masks import clean_load_mask
from cone_load.visualization import render_overlay


def example_masks() -> dict[str, np.ndarray]:
    masks = {}
    for name, apex in [('izquierda', 155), ('centrado', 320), ('derecha', 485)]:
        mask = np.zeros((400, 640), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array([[70, 330], [apex, 100], [570, 330]], np.int32)], 1)
        masks[name] = mask
    masks['no_evaluable'] = np.zeros((400, 640), dtype=np.uint8)
    return masks


def run_demo(output: Path) -> list[dict]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory is not empty: {output}. Choose a new directory.')
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / 'configs' / 'analytic_config.yaml')
    rows = []
    for expected, mask in example_masks().items():
        cleaned, warnings = clean_load_mask(mask, config)
        result = analyze_binary_mask(expected, cleaned, config, warnings=warnings)
        canvas = np.full((400, 640, 3), 235, dtype=np.uint8)
        canvas[mask > 0] = (125, 125, 125)
        render_overlay(canvas, cleaned, result, output / f'{expected}.png', config)
        rows.append({
            'example': expected,
            'expected': expected,
            'predicted': result.features['predicted_class'],
            'peak_x_norm': result.features['peak_x_norm'],
            'area_symmetry': result.features['area_symmetry'],
            'error_reason': result.features['error_reason'],
        })
    with (output / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'demo_synthetic')
    args = parser.parse_args()
    rows = run_demo(args.output_dir)
    for row in rows:
        print(f"{row['example']}: expected={row['expected']}, predicted={row['predicted']}")
    print('Synthetic software demonstration only; not a performance benchmark.')
    return 0 if all(r['expected'] == r['predicted'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
