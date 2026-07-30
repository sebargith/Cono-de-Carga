"""
Frame Hunter - Configuration Manager
=====================================
Configuración mínima para Nextcloud + extracción de frames.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        load_dotenv()
        self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        yc = data.get("nextcloud", {}) or {}
        self.nextcloud_url = os.getenv("NEXTCLOUD_URL") or yc.get("url", "")
        self.nextcloud_username = os.getenv("NEXTCLOUD_USERNAME") or yc.get("username", "")
        self.nextcloud_password = os.getenv("NEXTCLOUD_PASSWORD") or yc.get("password", "")

        missing = [k for k, v in {
            "NEXTCLOUD_URL": self.nextcloud_url,
            "NEXTCLOUD_USERNAME": self.nextcloud_username,
            "NEXTCLOUD_PASSWORD": self.nextcloud_password,
        }.items() if not v]
        if missing:
            raise ValueError(
                "Faltan credenciales de NextCloud. "
                f"Define {', '.join(missing)} en .env o en config/config.yaml"
            )

        # Nueva forma: una sola carpeta remota fija.
        # Compatibilidad hacia atrás: también acepta remote_folder al nivel raíz
        # y camera.base_path si el usuario reutiliza una config anterior.
        self.remote_folder = (
            yc.get("remote_folder")
            or data.get("remote_folder")
            or (data.get("camera", {}) or {}).get("base_path", "")
        ).strip().strip("/")

        if not self.remote_folder:
            raise ValueError(
                "Falta la carpeta remota de Nextcloud. "
                "Define nextcloud.remote_folder en config/config.yaml"
            )

        self.specific_videos = list(data.get("specific_videos", []) or [])

        lp = data.get("local_paths", {}) or {}
        self.videos_folder = lp.get("videos_folder", "data/videos")
        self.frames_root = lp.get("frames_root", "data/frames")

        ext = data.get("extraction", {}) or {}
        self.quality = int(ext.get("quality", 85))
        self.use_ffmpeg = bool(ext.get("use_ffmpeg", True))
        self.ffmpeg_qscale = int(ext.get("ffmpeg_qscale", 2))
        interval_raw = ext.get("interval_seconds", 0)
        try:
            self.interval_seconds = max(float(interval_raw or 0), 0.0)
        except (TypeError, ValueError):
            self.interval_seconds = 0.0

        Path(self.videos_folder).mkdir(parents=True, exist_ok=True)
        Path(self.frames_root).mkdir(parents=True, exist_ok=True)

    def get_remote_folder(self) -> str:
        return self.remote_folder

    def print_summary(self):
        print("🎯 FRAME HUNTER - NEXTCLOUD/FRAMES")
        print("=" * 50)
        print(f"📁 NextCloud: {self.nextcloud_url}")
        print(f"👤 Usuario: {self.nextcloud_username}")
        print(f"📂 Carpeta remota: {self.get_remote_folder()}")
        print(f"📋 Videos específicos: {len(self.specific_videos)}")
        print(f"📁 Videos locales: {self.videos_folder}")
        print(f"📂 Frames locales: {self.frames_root}")
        print(f"⏱️  Intervalo extracción: {self.interval_seconds}s")
        print(f"🎨 Calidad JPEG: {self.quality}")
        print(f"⚡ ffmpeg: {'sí' if self.use_ffmpeg else 'no'}")


def load_config() -> Config:
    return Config()
