from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cone_perception.io import get_nested, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida un modelo YOLO-seg entrenado.")
    parser.add_argument("--model", required=True, help="Ruta a best.pt o modelo YOLO.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "perception_config.yaml"),
        help="Archivo de configuracion YAML.",
    )
    parser.add_argument("--dataset-yaml", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("No esta instalado ultralytics.") from exc

    dataset_yaml = args.dataset_yaml or get_nested(
        config,
        ("dataset", "dataset_yaml"),
        str(ROOT / "data" / "yolo" / "carga_seg" / "dataset.yaml"),
    )
    model = YOLO(args.model)
    model.val(
        data=dataset_yaml,
        task="segment",
        imgsz=args.imgsz or int(get_nested(config, ("training", "imgsz"), 640)),
        device=args.device or get_nested(config, ("training", "device"), "0"),
        project=str(ROOT / "runs" / "val"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

