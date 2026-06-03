from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


class Config:
    """Configuration for Nextcloud video download and frame extraction."""

    def __init__(self, config_path: str | Path | None = None):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = (
            Path(config_path).resolve()
            if config_path
            else self.project_root / "config" / "config.yaml"
        )

        load_dotenv(self.project_root / ".env")
        self._load_config()

    def _load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(
                "Archivo de configuracion no encontrado: "
                f"{self.config_path}. Copia config/config.example.yaml como config/config.yaml."
            )

        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        nextcloud_cfg = data.get("nextcloud", {}) or {}
        self.nextcloud_url = os.getenv("NEXTCLOUD_URL") or nextcloud_cfg.get("url", "")
        self.nextcloud_username = (
            os.getenv("NEXTCLOUD_USERNAME") or nextcloud_cfg.get("username", "")
        )
        self.nextcloud_password = (
            os.getenv("NEXTCLOUD_PASSWORD") or nextcloud_cfg.get("password", "")
        )

        missing = [
            key
            for key, value in {
                "NEXTCLOUD_URL": self.nextcloud_url,
                "NEXTCLOUD_USERNAME": self.nextcloud_username,
                "NEXTCLOUD_PASSWORD": self.nextcloud_password,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Faltan credenciales de Nextcloud. Define "
                f"{', '.join(missing)} en .env o en config/config.yaml."
            )

        self.remote_folder = (
            nextcloud_cfg.get("remote_folder")
            or data.get("remote_folder")
            or (data.get("camera", {}) or {}).get("base_path", "")
        ).strip().strip("/")
        if not self.remote_folder:
            raise ValueError(
                "Falta la carpeta remota. Define nextcloud.remote_folder en config/config.yaml."
            )

        self.specific_videos = list(data.get("specific_videos", []) or [])

        local_paths = data.get("local_paths", {}) or {}
        self.videos_folder = self._resolve_local_path(
            local_paths.get("videos_folder", "data/videos")
        )
        self.frames_root = self._resolve_local_path(
            local_paths.get("frames_root", "data/frames")
        )

        extraction = data.get("extraction", {}) or {}
        self.quality = int(extraction.get("quality", 85))
        self.use_ffmpeg = bool(extraction.get("use_ffmpeg", True))
        self.ffmpeg_qscale = int(extraction.get("ffmpeg_qscale", 2))
        self.interval_seconds = self._parse_interval(
            extraction.get("interval_seconds", 0)
        )

        self.videos_folder.mkdir(parents=True, exist_ok=True)
        self.frames_root.mkdir(parents=True, exist_ok=True)

    def _resolve_local_path(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.project_root / path

    @staticmethod
    def _parse_interval(value: object) -> float:
        try:
            return max(float(value or 0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_remote_folder(self) -> str:
        return self.remote_folder

    def print_summary(self) -> None:
        print("FRAME HUNTER - NEXTCLOUD/FRAMES")
        print("=" * 50)
        print(f"Nextcloud: {self.nextcloud_url}")
        print(f"Usuario: {self.nextcloud_username}")
        print(f"Carpeta remota: {self.get_remote_folder()}")
        print(f"Videos especificos: {len(self.specific_videos)}")
        print(f"Videos locales: {self.videos_folder}")
        print(f"Frames locales: {self.frames_root}")
        print(f"Intervalo extraccion: {self.interval_seconds}s")
        print(f"Calidad JPEG: {self.quality}")
        print(f"ffmpeg: {'si' if self.use_ffmpeg else 'no'}")


def load_config(config_path: str | Path | None = None) -> Config:
    return Config(config_path=config_path)
