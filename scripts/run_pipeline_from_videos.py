from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    "perception/models/reference/"
    "carga_visible_yolov8n-seg_20260527_195509/best.pt"
)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline desde videos locales: extrae frames, genera mascaras YOLO "
            "compatibles con el analitico y corre el analitico geometrico."
        )
    )
    parser.add_argument("--videos-dir", required=True, help="Carpeta con videos de entrada.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre del experimento. Por defecto usa timestamp.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Carpeta de trabajo visible para Docker. Debe quedar dentro de "
            "perception/model_testing/pipeline_runs/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Carpeta final de resultados. Debe quedar dentro de "
            "outputs/pipeline_runs/."
        ),
    )
    parser.add_argument(
        "--perception-dir",
        default="perception",
        help="Directorio del modulo de percepcion con docker-compose.yml.",
    )
    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_MODEL,
        help="Ruta al modelo YOLO-seg. Debe quedar dentro del directorio perception.",
    )
    parser.add_argument(
        "--analytic-config",
        default="configs/analytic_config.yaml",
        help="Configuracion del analitico geometrico.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="Extrae 1 frame cada N segundos. Usa 0 para extraer todos los frames.",
    )
    parser.add_argument("--jpg-quality", type=int, default=90)
    parser.add_argument("--recursive", action="store_true", help="Buscar videos recursivamente.")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--mask-mode",
        choices=("largest", "merge_all"),
        default="largest",
        help="Como combinar instancias YOLO antes de pasar al analitico.",
    )
    parser.add_argument(
        "--docker-service",
        default="cone-yolo",
        help="Servicio Docker Compose usado para correr YOLO.",
    )
    parser.add_argument(
        "--start-docker",
        action="store_true",
        help="Ejecuta docker compose up -d antes de correr YOLO.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Borra la corrida previa con el mismo run-name si existe.",
    )
    return parser.parse_args()


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def require_dir(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"No existe {name}: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} no es un directorio: {path}")
    return path


def require_file(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"No existe {name}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} no es un archivo: {path}")
    return path


def prepare_clean_dir(path: Path, allowed_base: Path, overwrite: bool, label: str) -> Path:
    path = path.resolve()
    allowed_base = allowed_base.resolve()
    if not is_relative_to(path, allowed_base) or path == allowed_base:
        raise ValueError(f"{label} debe quedar dentro de {allowed_base}: {path}")

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{label} existe pero no es un directorio: {path}")

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{label} ya existe y no esta vacio: {path}. Usa --overwrite "
                "o elige otro --run-name."
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return path


def default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def list_videos(videos_dir: Path, recursive: bool) -> list[Path]:
    iterator = videos_dir.rglob("*") if recursive else videos_dir.glob("*")
    allowed = {ext.lower() for ext in VIDEO_EXTENSIONS}
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in allowed)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "video"


def video_id(video_path: Path, videos_dir: Path) -> str:
    try:
        rel = video_path.relative_to(videos_dir)
        stem = "__".join(rel.with_suffix("").parts)
    except ValueError:
        stem = video_path.stem
    return safe_name(stem)


def frame_step_from_interval(fps: float, interval_seconds: float) -> int:
    if interval_seconds <= 0:
        return 1
    if fps <= 0:
        return 1
    return max(int(round(fps * interval_seconds)), 1)


def write_manifest_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_frames(
    videos: list[Path],
    videos_dir: Path,
    frames_dir: Path,
    interval_seconds: float,
    jpg_quality: int,
    max_frames_per_video: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
    used_frame_names: set[str] = set()

    for idx, video_path in enumerate(videos, start=1):
        cap = cv2.VideoCapture(str(video_path))
        current_video_id = video_id(video_path, videos_dir)
        if not cap.isOpened():
            video_rows.append(
                {
                    "video_name": video_path.name,
                    "video_path": str(video_path),
                    "video_id": current_video_id,
                    "fps": "",
                    "total_frames": "",
                    "saved_frames": 0,
                    "status": "error_no_se_pudo_abrir_video",
                }
            )
            print(f"[{idx}/{len(videos)}] No se pudo abrir: {video_path}")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = frame_step_from_interval(fps, interval_seconds)
        frame_index = 0
        saved = 0

        print(f"[{idx}/{len(videos)}] Extrayendo frames: {video_path.name}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            should_save = interval_seconds <= 0 or frame_index % step == 0
            if should_save:
                frame_name = f"{current_video_id}__frame{frame_index:06d}.jpg"
                if frame_name in used_frame_names:
                    raise ValueError(f"Nombre de frame duplicado: {frame_name}")
                used_frame_names.add(frame_name)

                out_path = frames_dir / frame_name
                if not cv2.imwrite(str(out_path), frame, encode_param):
                    cap.release()
                    raise OSError(f"No se pudo escribir frame: {out_path}")

                timestamp_sec = frame_index / fps if fps > 0 else ""
                frame_rows.append(
                    {
                        "video_name": video_path.name,
                        "video_path": str(video_path),
                        "video_id": current_video_id,
                        "frame_name": frame_name,
                        "frame_path": str(out_path),
                        "frame_index": frame_index,
                        "timestamp_sec": (
                            "" if timestamp_sec == "" else f"{timestamp_sec:.6f}"
                        ),
                    }
                )
                saved += 1
                if max_frames_per_video is not None and saved >= max_frames_per_video:
                    break

            frame_index += 1

        cap.release()
        video_rows.append(
            {
                "video_name": video_path.name,
                "video_path": str(video_path),
                "video_id": current_video_id,
                "fps": f"{fps:.6f}" if fps > 0 else "",
                "total_frames": total_frames,
                "saved_frames": saved,
                "status": "ok" if saved > 0 else "sin_frames_extraidos",
            }
        )
        print(f"  Frames guardados: {saved}")

    return frame_rows, video_rows


def perception_host_path(value: str | Path, perception_dir: Path) -> Path:
    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([ROOT / raw, perception_dir / raw])

    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            if is_relative_to(resolved, perception_dir):
                return resolved

    candidate_list = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No se pudo resolver la ruta dentro de perception. "
        f"Valor: {value}. Candidatos: {candidate_list}"
    )


def to_container_path(host_path: Path, perception_dir: Path) -> str:
    resolved = host_path.resolve()
    perception_root = perception_dir.resolve()
    if not is_relative_to(resolved, perception_root):
        raise ValueError(f"La ruta no esta montada en Docker: {resolved}")
    rel = resolved.relative_to(perception_root)
    return "/workspace/" + rel.as_posix()


def run_command(command: list[str], cwd: Path, label: str) -> None:
    print(f"\n== {label} ==")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"Fallo {label} con codigo {completed.returncode}")


def run_yolo_masks(
    perception_dir: Path,
    docker_service: str,
    yolo_model: Path,
    frames_dir: Path,
    masks_dir: Path,
    conf: float,
    imgsz: int | None,
    device: str,
    mask_mode: str,
    start_docker: bool,
) -> None:
    if start_docker:
        run_command(["docker", "compose", "up", "-d"], perception_dir, "levantar Docker")

    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        docker_service,
        "python",
        "scripts/predict_to_analytic_masks.py",
        "--model",
        to_container_path(yolo_model, perception_dir),
        "--images-dir",
        to_container_path(frames_dir, perception_dir),
        "--masks-dir",
        to_container_path(masks_dir, perception_dir),
        "--conf",
        str(conf),
        "--device",
        str(device),
        "--mask-mode",
        mask_mode,
        "--overwrite",
    ]
    if imgsz is not None:
        command.extend(["--imgsz", str(imgsz)])

    run_command(command, perception_dir, "YOLO a mascaras analiticas")


def run_analytic(frames_dir: Path, masks_dir: Path, analytic_dir: Path, config_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_simple_analytic.py"),
        "--images-dir",
        str(frames_dir),
        "--masks-dir",
        str(masks_dir),
        "--output-dir",
        str(analytic_dir),
        "--config",
        str(config_path),
    ]
    run_command(command, ROOT, "analitico geometrico")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_features(features_csv: Path) -> dict[str, Any]:
    rows = read_csv_rows(features_csv)
    class_counts = Counter(row.get("predicted_class", "") for row in rows)
    valid_rows = [row for row in rows if row.get("is_valid") == "True"]
    invalid_reasons = Counter(
        row.get("error_reason", "") for row in rows if row.get("is_valid") != "True"
    )
    return {
        "rows": len(rows),
        "valid": len(valid_rows),
        "invalid": len(rows) - len(valid_rows),
        "predicted_class_counts": dict(class_counts),
        "invalid_reason_counts": dict(invalid_reasons),
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    run_name = safe_name(args.run_name or default_run_name())

    videos_dir = require_dir(resolve_repo_path(args.videos_dir), "videos-dir")
    perception_dir = require_dir(resolve_repo_path(args.perception_dir), "perception-dir")
    analytic_config = require_file(resolve_repo_path(args.analytic_config), "analytic-config")
    yolo_model = require_file(
        perception_host_path(args.yolo_model, perception_dir),
        "yolo-model",
    )

    work_base = perception_dir / "model_testing" / "pipeline_runs"
    output_base = ROOT / "outputs" / "pipeline_runs"
    work_dir = resolve_repo_path(args.work_dir) if args.work_dir else work_base / run_name
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else output_base / run_name

    work_dir = prepare_clean_dir(work_dir, work_base, args.overwrite, "work-dir")
    output_dir = prepare_clean_dir(output_dir, output_base, args.overwrite, "output-dir")

    frames_dir = work_dir / "frames"
    masks_dir = work_dir / "yolo_masks"
    analytic_dir = output_dir / "analytic"
    manifests_dir = output_dir / "manifests"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    analytic_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(videos_dir, recursive=args.recursive)
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]
    if not videos:
        raise FileNotFoundError(f"No se encontraron videos en: {videos_dir}")

    frame_rows, video_rows = extract_frames(
        videos=videos,
        videos_dir=videos_dir,
        frames_dir=frames_dir,
        interval_seconds=max(float(args.interval_seconds), 0.0),
        jpg_quality=args.jpg_quality,
        max_frames_per_video=args.max_frames_per_video,
    )
    if not frame_rows:
        raise RuntimeError("No se extrajo ningun frame; no se puede continuar.")

    frame_manifest = manifests_dir / "frames.csv"
    video_manifest = manifests_dir / "videos.csv"
    write_manifest_csv(
        frame_manifest,
        frame_rows,
        [
            "video_name",
            "video_path",
            "video_id",
            "frame_name",
            "frame_path",
            "frame_index",
            "timestamp_sec",
        ],
    )
    write_manifest_csv(
        video_manifest,
        video_rows,
        ["video_name", "video_path", "video_id", "fps", "total_frames", "saved_frames", "status"],
    )

    run_yolo_masks(
        perception_dir=perception_dir,
        docker_service=args.docker_service,
        yolo_model=yolo_model,
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        mask_mode=args.mask_mode,
        start_docker=args.start_docker,
    )

    copy_if_exists(masks_dir / "manifest.csv", manifests_dir / "yolo_manifest.csv")
    copy_if_exists(masks_dir / "summary.json", manifests_dir / "yolo_summary.json")

    run_analytic(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        analytic_dir=analytic_dir,
        config_path=analytic_config,
    )

    features_csv = analytic_dir / "tables" / "features.csv"
    summary = {
        "run_name": run_name,
        "videos_dir": str(videos_dir),
        "work_dir": str(work_dir),
        "output_dir": str(output_dir),
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "analytic_dir": str(analytic_dir),
        "video_count": len(videos),
        "frame_count": len(frame_rows),
        "yolo_model": str(yolo_model),
        "conf": args.conf,
        "device": args.device,
        "mask_mode": args.mask_mode,
        "interval_seconds": max(float(args.interval_seconds), 0.0),
        "features_csv": str(features_csv),
        "feature_summary": summarize_features(features_csv),
    }
    summary_path = output_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nPipeline completado")
    print(f"Frames: {frames_dir}")
    print(f"Mascaras YOLO: {masks_dir}")
    print(f"Resultados analitico: {analytic_dir}")
    print(f"Resumen: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
