from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .run_storage import PROJECT_ROOT, RunPaths


@dataclass(frozen=True)
class PipelineParams:
    interval_seconds: float
    max_frames_per_video: int | None
    conf: float
    device: str
    start_docker: bool
    video_min_valid_ratio: float
    video_majority_ratio: float


def build_command(paths: RunPaths, params: PipelineParams) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline_from_videos.py"),
        "--videos-dir",
        str(paths.videos_dir),
        "--run-name",
        paths.run_name,
        "--work-dir",
        str(paths.work_dir),
        "--output-dir",
        str(paths.result_dir),
        "--interval-seconds",
        str(params.interval_seconds),
        "--conf",
        str(params.conf),
        "--device",
        str(params.device),
        "--video-min-valid-ratio",
        str(params.video_min_valid_ratio),
        "--video-majority-ratio",
        str(params.video_majority_ratio),
        "--overwrite",
    ]
    if params.max_frames_per_video is not None:
        command.extend(["--max-frames-per-video", str(params.max_frames_per_video)])
    if params.start_docker:
        command.append("--start-docker")
    return command


def run_pipeline(paths: RunPaths, params: PipelineParams) -> tuple[int, str]:
    command = build_command(paths, params)
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = "\n".join(
        part
        for part in [
            " ".join(command),
            completed.stdout,
            completed.stderr,
        ]
        if part
    )
    return completed.returncode, log


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)
