from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cone_perception.io import get_nested, load_yaml


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corre inferencia YOLO-seg sobre imagenes o video.")
    parser.add_argument("--model", required=True, help="Ruta a best.pt o modelo YOLO.")
    parser.add_argument("--source", required=True, help="Imagen, carpeta o video.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "perception_config.yaml"),
        help="Archivo de configuracion YAML.",
    )
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument(
        "--project",
        default=str(ROOT / "model_testing" / "outputs"),
        help="Directorio donde se guardan las predicciones.",
    )
    parser.add_argument("--name", default="predict", help="Nombre de la corrida de prediccion.")
    parser.add_argument("--exist-ok", action="store_true", help="Permite reutilizar una carpeta de salida existente.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("No esta instalado ultralytics.") from exc

    model = YOLO(args.model)
    project = resolve_repo_path(args.project)
    model.predict(
        source=args.source,
        task="segment",
        conf=args.conf if args.conf is not None else float(get_nested(config, ("prediction", "conf"), 0.35)),
        imgsz=args.imgsz or int(get_nested(config, ("training", "imgsz"), 640)),
        device=args.device or get_nested(config, ("training", "device"), "0"),
        project=str(project),
        name=args.name,
        exist_ok=args.exist_ok,
        save=True,
        save_txt=args.save_txt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
