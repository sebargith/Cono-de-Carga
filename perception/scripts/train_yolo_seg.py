from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cone_perception.io import ensure_dir, get_nested, load_yaml, write_yaml


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
    return flat


def sanitize_metric_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_.\-/ ]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("_") or "metric"


def read_results_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if key is None or value is None or value == "":
                    continue
                try:
                    parsed[sanitize_metric_name(key)] = float(value)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def log_mlflow_metrics_from_results(mlflow_module: Any, results_csv: Path) -> None:
    rows = read_results_csv(results_csv)
    if not rows:
        return

    last = rows[-1]
    for key, value in last.items():
        mlflow_module.log_metric(f"final/{key}", value)

    by_key: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            by_key.setdefault(key, []).append(value)
    for key, values in by_key.items():
        if values:
            mlflow_module.log_metric(f"best/{key}", max(values))


def setup_mlflow(config: dict[str, Any], run_name: str):
    if not bool(get_nested(config, ("mlflow", "enabled"), True)):
        return None
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow esta habilitado, pero no esta instalado. "
            "Reconstruye la imagen Docker para instalar requirements/docker.txt."
        ) from exc

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or str(
        get_nested(config, ("mlflow", "tracking_uri"), "file:/workspace/mlruns")
    )
    experiment_name = str(
        get_nested(config, ("mlflow", "experiment_name"), "cono_carga_yolo_seg")
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")
    print(f"MLflow run_id: {run.info.run_id}")
    return mlflow


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def disable_ultralytics_auto_mlflow() -> None:
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"mlflow": False})
    except Exception as exc:
        print(f"Advertencia: no se pudo desactivar MLflow automatico de Ultralytics: {exc}")


def get_actual_run_dir(model: Any, train_result: Any, fallback: Path, run_name: str) -> Path:
    candidates: list[Path] = []

    trainer = getattr(model, "trainer", None)
    trainer_save_dir = getattr(trainer, "save_dir", None)
    if trainer_save_dir:
        candidates.append(Path(trainer_save_dir))

    result_save_dir = getattr(train_result, "save_dir", None)
    if result_save_dir:
        candidates.append(Path(result_save_dir))

    candidates.append(fallback)
    candidates.extend((ROOT / "runs").rglob(run_name))

    for candidate in candidates:
        if (candidate / "weights" / "best.pt").exists():
            return candidate

    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLO para segmentar carga visible.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "perception_config.yaml"),
        help="Archivo de configuracion YAML.",
    )
    parser.add_argument("--dataset-yaml", default=None, help="Ruta a dataset.yaml YOLO.")
    parser.add_argument(
        "--model",
        default=None,
        help="Arquitectura YOLO o pesos iniciales. Para entrenar desde cero usar .yaml, ej: yolov8n-seg.yaml.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Usar pesos preentrenados. Por defecto viene desde perception_config.yaml.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "No esta instalado ultralytics. Instala dependencias con: "
            "pip install -r perception/requirements/perception.txt"
        ) from exc

    dataset_yaml = Path(
        args.dataset_yaml
        or get_nested(
            config,
            ("dataset", "dataset_yaml"),
            str(ROOT / "data" / "yolo" / "carga_seg" / "dataset.yaml"),
        )
    )
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"No existe dataset.yaml: {dataset_yaml}. Primero ejecuta prepare_yolo_dataset.py"
        )

    model_name = args.model or str(get_nested(config, ("training", "model"), "yolov8n-seg.yaml"))
    pretrained = args.pretrained
    if pretrained is None:
        pretrained = bool(get_nested(config, ("training", "pretrained"), False))

    model_suffix = Path(model_name).suffix.lower()
    if model_suffix == ".pt" and not pretrained:
        raise ValueError(
            "Se pidio pretrained=false, pero --model apunta a un .pt. "
            "Un .pt ya contiene pesos entrenados. Para entrenar desde cero usa una arquitectura .yaml "
            "como yolov8n-seg.yaml; para fine-tuning usa --pretrained."
        )

    run_name = args.run_name or str(
        get_nested(
            config,
            ("training", "run_name"),
            f"carga_visible_{Path(model_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    )

    project = resolve_repo_path(get_nested(config, ("training", "project"), ROOT / "runs" / "train"))
    models_dir = ensure_dir(resolve_repo_path(get_nested(config, ("training", "models_dir"), ROOT / "models")))

    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": args.epochs or int(get_nested(config, ("training", "epochs"), 100)),
        "batch": args.batch or int(get_nested(config, ("training", "batch"), 8)),
        "imgsz": args.imgsz or int(get_nested(config, ("training", "imgsz"), 640)),
        "device": args.device or get_nested(config, ("training", "device"), "0"),
        "project": str(project),
        "name": run_name,
        "task": "segment",
        "patience": int(get_nested(config, ("training", "patience"), 30)),
        "workers": int(get_nested(config, ("training", "workers"), 4)),
        "optimizer": get_nested(config, ("training", "optimizer"), "auto"),
        "lr0": float(get_nested(config, ("training", "lr0"), 0.001)),
        "weight_decay": float(get_nested(config, ("training", "weight_decay"), 0.0005)),
        "degrees": float(get_nested(config, ("augmentation", "degrees"), 0.0)),
        "translate": float(get_nested(config, ("augmentation", "translate"), 0.1)),
        "scale": float(get_nested(config, ("augmentation", "scale"), 0.5)),
        "shear": float(get_nested(config, ("augmentation", "shear"), 0.0)),
        "flipud": float(get_nested(config, ("augmentation", "flipud"), 0.0)),
        "fliplr": float(get_nested(config, ("augmentation", "fliplr"), 0.5)),
        "mosaic": float(get_nested(config, ("augmentation", "mosaic"), 0.5)),
        "amp": bool(get_nested(config, ("training", "amp"), True)),
        "exist_ok": bool(get_nested(config, ("training", "exist_ok"), False)),
    }
    if model_suffix in {".yaml", ".yml"}:
        train_kwargs["pretrained"] = pretrained

    freeze = get_nested(config, ("training", "freeze"), None)
    if freeze is not None:
        train_kwargs["freeze"] = freeze

    mlflow = setup_mlflow(config, run_name)
    if mlflow is not None:
        params = {
            "model_name": model_name,
            "model_suffix": model_suffix,
            "pretrained": pretrained,
            "dataset_yaml": str(dataset_yaml),
            "run_name": run_name,
        }
        params.update({f"train.{k}": v for k, v in train_kwargs.items() if isinstance(v, (str, int, float, bool))})
        params.update({f"config.{k}": v for k, v in flatten_dict(config).items()})
        mlflow.log_params(params)
        mlflow.log_artifact(str(args.config), artifact_path="config")
        mlflow.log_artifact(str(dataset_yaml), artifact_path="dataset")
        manifest = dataset_yaml.parent / "manifest.csv"
        if manifest.exists():
            mlflow.log_artifact(str(manifest), artifact_path="dataset")

    disable_ultralytics_auto_mlflow()
    model = YOLO(model_name)
    print(f"Entrenando {model_name}")
    print(f"Pretrained: {pretrained}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Run: {project / run_name}")
    try:
        train_result = model.train(**train_kwargs)
    except Exception:
        if mlflow is not None:
            mlflow.set_tag("status", "failed")
        raise

    run_dir = get_actual_run_dir(model, train_result, project / run_name, run_name)
    best = run_dir / "weights" / "best.pt"
    exported_best = None
    if best.exists():
        export_dir = ensure_dir(models_dir / run_name)
        exported_best = export_dir / "best.pt"
        shutil.copy2(best, exported_best)
        last = run_dir / "weights" / "last.pt"
        if last.exists():
            shutil.copy2(last, export_dir / "last.pt")
        write_yaml(export_dir / "train_config_used.yaml", config)
        print(f"Modelo copiado a: {export_dir / 'best.pt'}")
    else:
        print(f"Entrenamiento terminado, pero no se encontro best.pt en {best}")

    if mlflow is not None:
        results_csv = run_dir / "results.csv"
        log_mlflow_metrics_from_results(mlflow, results_csv)
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv), artifact_path="training")
        args_yaml = run_dir / "args.yaml"
        if args_yaml.exists():
            mlflow.log_artifact(str(args_yaml), artifact_path="training")
        weight_logging = bool(get_nested(config, ("mlflow", "log_weight_artifacts"), True))
        if weight_logging:
            if exported_best is not None and exported_best.exists():
                mlflow.log_artifact(str(exported_best), artifact_path="weights")
            exported_last = models_dir / run_name / "last.pt"
            if exported_last.exists():
                mlflow.log_artifact(str(exported_last), artifact_path="weights")
        if bool(get_nested(config, ("mlflow", "log_run_dir_artifacts"), False)) and run_dir.exists():
            mlflow.log_artifacts(str(run_dir), artifact_path="ultralytics_run")
        mlflow.set_tag("status", "completed")
        mlflow.end_run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
