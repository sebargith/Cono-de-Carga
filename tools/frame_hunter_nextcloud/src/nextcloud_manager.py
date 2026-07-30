"""
Frame Hunter - NextCloud Manager
=================================
Gestor de acceso, descarga y extracción de frames desde NextCloud.
"""

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote

import cv2
import requests
from requests.auth import HTTPBasicAuth

from src.config import Config


class NextCloudManager:
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.nextcloud_url.rstrip("/")
        self.username = config.nextcloud_username
        self.webdav_url = f"{self.base_url}/remote.php/dav/files/{self.username}/"
        self.auth = HTTPBasicAuth(config.nextcloud_username, config.nextcloud_password)

    def _remote_folder(self) -> str:
        return self.config.get_remote_folder().strip().strip("/")

    @staticmethod
    def _normalize_name(name: str) -> str:
        return Path(name.strip()).stem.lower()

    def _resolve_requested_video(self, requested: str, available_videos: List[str]) -> Optional[str]:
        requested_clean = requested.strip()
        if not requested_clean:
            return None

        available_map_exact = {video.lower(): video for video in available_videos}
        exact = available_map_exact.get(requested_clean.lower())
        if exact:
            return exact

        requested_stem = self._normalize_name(requested_clean)
        stem_matches = [video for video in available_videos if self._normalize_name(video) == requested_stem]

        if len(stem_matches) == 1:
            return stem_matches[0]
        return None

    def test_connection(self) -> bool:
        print("🔧 Probando conexión a NextCloud")
        remote_folder = self._remote_folder()
        print(f"📂 Carpeta: {remote_folder}")

        try:
            webdav_url = f"{self.webdav_url}{remote_folder}/"
            response = requests.request(
                method="PROPFIND",
                url=webdav_url,
                auth=self.auth,
                headers={"Depth": "0"},
                timeout=15,
            )

            if response.status_code == 207:
                return True
            if response.status_code == 404:
                print("❌ Carpeta remota no encontrada")
                return False
            if response.status_code == 401:
                print("❌ Credenciales incorrectas")
                return False

            print(f"❌ Error HTTP: {response.status_code}")
            return False

        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False

    def get_all_available_videos(self) -> List[str]:
        remote_folder = self._remote_folder()
        print(f"🔍 Obteniendo videos desde {remote_folder}")

        try:
            webdav_url = f"{self.webdav_url}{remote_folder}/"
            response = requests.request(
                method="PROPFIND",
                url=webdav_url,
                auth=self.auth,
                headers={"Depth": "1"},
                timeout=15,
            )

            if response.status_code != 207:
                print(f"❌ Error obteniendo listado: {response.status_code}")
                return []

            content = response.text
            href_pattern = r"<d:href>/remote\.php/dav/files/[^/]+/.*?/([^/<]+)</d:href>"
            matches = re.findall(href_pattern, content, re.IGNORECASE)

            videos = []
            for match in matches:
                decoded_name = unquote(match)
                ext = Path(decoded_name).suffix.lower()
                if ext in self.VIDEO_EXTENSIONS:
                    videos.append(decoded_name)

            return sorted(set(videos))

        except Exception as e:
            print(f"❌ Error listando videos: {e}")
            return []

    def validate_specific_videos(self, specific_videos: Optional[List[str]] = None) -> List[str]:
        videos_to_validate = specific_videos or self.config.specific_videos
        all_available = self.get_all_available_videos()

        if not all_available:
            print("❌ No se pudo obtener la lista de videos del servidor")
            return []

        valid_videos = []
        invalid_videos = []

        for requested in videos_to_validate:
            resolved = self._resolve_requested_video(requested, all_available)
            if resolved:
                valid_videos.append(resolved)
                if resolved == requested:
                    print(f"  ✅ {requested}")
                else:
                    print(f"  ✅ {requested} -> {resolved}")
            else:
                invalid_videos.append(requested)
                print(f"  ❌ {requested}")

        if invalid_videos:
            print(f"\n⚠️  No encontrados: {len(invalid_videos)}")

        return valid_videos

    def download_video(
        self,
        video_filename: str,
        videos_folder: Optional[str] = None,
    ) -> Optional[str]:
        dest_folder = videos_folder or self.config.videos_folder
        remote_folder = self._remote_folder()

        video_path = f"{remote_folder}/{video_filename}"
        url = f"{self.webdav_url}{video_path}"

        print(f"📥 Descargando: {video_filename}")

        try:
            response = requests.get(url, auth=self.auth, stream=True, timeout=60)
            response.raise_for_status()

            Path(dest_folder).mkdir(parents=True, exist_ok=True)
            local_video_path = os.path.join(dest_folder, video_filename)

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(local_video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        print(
                            f"\r📥 {progress:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                            end="",
                            flush=True,
                        )

            if total_size > 0:
                print()

            print(f"💾 Guardado en: {local_video_path}")
            return local_video_path

        except Exception as e:
            print(f"❌ Error descargando {video_filename}: {e}")
            return None

    def extract_all_frames_from_video(
        self,
        video_filename: str,
        video_path: str,
        quality: Optional[int] = None,
        interval_seconds: Optional[float] = None,
    ) -> tuple[bool, str, int]:
        video_name = os.path.splitext(os.path.basename(video_filename))[0]
        frames_folder = os.path.join(self.config.frames_root, video_name)
        Path(frames_folder).mkdir(parents=True, exist_ok=True)

        jpeg_quality = int(quality or self.config.quality)
        try:
            interval = max(
                float(interval_seconds if interval_seconds is not None else self.config.interval_seconds),
                0.0,
            )
        except (TypeError, ValueError):
            interval = 0.0

        if self.config.use_ffmpeg:
            out_pattern = os.path.join(frames_folder, f"{video_name}_frame%06d.jpg")
            q = int(getattr(self.config, "ffmpeg_qscale", 2))
            cmd_parts = [
                "ffmpeg -y -loglevel error -stats",
                f"-i {shlex.quote(video_path)}",
                "-map 0:v:0",
                "-vsync 0",
            ]
            if interval > 0:
                cmd_parts.append(f"-vf fps=1/{interval}")
            cmd_parts.extend([
                "-start_number 0",
                f"-q:v {q}",
                shlex.quote(out_pattern),
            ])
            cmd = " ".join(cmd_parts)

            try:
                subprocess.run(cmd, shell=True, check=True)
                n = len(list(Path(frames_folder).glob(f"{video_name}_frame*.jpg")))
                print(f"✅ Extraídos {n} frames con ffmpeg → {frames_folder}")
                return True, frames_folder, n
            except Exception as e:
                print(f"⚠️ ffmpeg falló, usando OpenCV. Detalle: {e}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ No se pudo abrir el video")
            return False, frames_folder, 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_interval = 1
        if interval > 0 and fps > 0:
            frame_interval = max(int(round(fps * interval)), 1)

        i = 0
        saved = 0
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if interval == 0 or i % frame_interval == 0:
                out_path = os.path.join(frames_folder, f"{video_name}_frame{i:06d}.jpg")
                if not cv2.imwrite(out_path, frame, encode_param):
                    cap.release()
                    print(f"❌ Error guardando: {out_path}")
                    return False, frames_folder, saved
                saved += 1

                if saved % 500 == 0:
                    print(f"  ... {saved} frames guardados (leídos {i + 1}/{total if total > 0 else '?'})")
            i += 1

        cap.release()
        print(f"✅ Extraídos {saved} frames con OpenCV → {frames_folder}")
        return True, frames_folder, saved

    def process_specific_videos(self) -> dict:
        valid_videos = self.validate_specific_videos()

        if not valid_videos:
            print("❌ No hay videos válidos para procesar")
            return {"success": 0, "total": 0, "frames_extracted": 0}

        print(f"\n🎯 Se procesarán {len(valid_videos)} videos")
        print(f"⏱️  Intervalo: {self.config.interval_seconds}s")
        print(f"🎨 Calidad: {self.config.quality}")
        print(f"⚡ ffmpeg: {'sí' if self.config.use_ffmpeg else 'no'}")
        print()

        answer = input("¿Continuar? (y/N): ").strip().lower()
        if answer not in {"y", "yes", "s", "si", "sí"}:
            print("❌ Procesamiento cancelado")
            return {"success": 0, "total": 0, "frames_extracted": 0}

        success_count = 0
        total_frames_extracted = 0

        for idx, video in enumerate(valid_videos, 1):
            print(f"\n[{idx}/{len(valid_videos)}] 🎬 {video}")
            local_video_path = self.download_video(video)
            if not local_video_path:
                continue

            ok, frames_folder, n_frames = self.extract_all_frames_from_video(video, local_video_path)
            if ok:
                success_count += 1
                total_frames_extracted += n_frames
                print(f"📁 Frames en: {frames_folder}")

        return {
            "success": success_count,
            "total": len(valid_videos),
            "frames_extracted": total_frames_extracted,
        }
