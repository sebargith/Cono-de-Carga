from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WEB_ROOT.parent
UPLOADS_ROOT = WEB_ROOT / "data" / "uploads"
RUNS_ROOT = WEB_ROOT / "outputs" / "runs"


@dataclass(frozen=True)
class RunPaths:
    run_name: str
    upload_dir: Path
    videos_dir: Path
    run_dir: Path
    work_dir: Path
    result_dir: Path
    analytic_dir: Path
    overlays_dir: Path
    features_csv: Path
    frame_manifest_csv: Path
    video_summary_csv: Path
    indicator_summary_csv: Path
    indicator_distributions_csv: Path
    pipeline_summary_json: Path


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or default_run_name()


def default_run_name() -> str:
    return datetime.now().strftime("web_%Y%m%d_%H%M%S")


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def remove_inside(path: Path, allowed_root: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    root = allowed_root.resolve()
    if not is_relative_to(resolved, root) or resolved == root:
        raise ValueError(f"Ruta fuera del area permitida: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def paths_for_run(run_name: str) -> RunPaths:
    clean = safe_name(run_name)
    upload_dir = UPLOADS_ROOT / clean
    videos_dir = upload_dir / "videos"
    run_dir = RUNS_ROOT / clean
    work_dir = run_dir / "work"
    result_dir = run_dir / "results"
    analytic_dir = result_dir / "analytic"
    return RunPaths(
        run_name=clean,
        upload_dir=upload_dir,
        videos_dir=videos_dir,
        run_dir=run_dir,
        work_dir=work_dir,
        result_dir=result_dir,
        analytic_dir=analytic_dir,
        overlays_dir=analytic_dir / "overlays",
        features_csv=analytic_dir / "tables" / "features.csv",
        frame_manifest_csv=result_dir / "manifests" / "frames.csv",
        video_summary_csv=result_dir / "video_summary.csv",
        indicator_summary_csv=result_dir / "statistics" / "indicator_summary.csv",
        indicator_distributions_csv=(
            result_dir / "statistics" / "indicator_distributions.csv"
        ),
        pipeline_summary_json=result_dir / "pipeline_summary.json",
    )


def list_runs() -> list[str]:
    if not RUNS_ROOT.exists():
        return []
    return sorted(
        [path.name for path in RUNS_ROOT.iterdir() if path.is_dir()],
        reverse=True,
    )


def prepare_uploads(paths: RunPaths, uploaded_files: list) -> int:
    remove_inside(paths.upload_dir, UPLOADS_ROOT)
    paths.videos_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for uploaded in uploaded_files:
        target = paths.videos_dir / safe_name(uploaded.name)
        with target.open("wb") as f:
            f.write(uploaded.getbuffer())
        count += 1
    return count


def overlay_paths(paths: RunPaths) -> list[Path]:
    if not paths.overlays_dir.exists():
        return []
    return sorted(paths.overlays_dir.glob("*.jpg"))


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())
