from __future__ import annotations

import os
import shlex
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, unquote, urlsplit

import cv2
import requests
from requests.auth import HTTPBasicAuth

from src.config import Config


class NextCloudManager:
    """Access Nextcloud videos through WebDAV and extract image frames."""

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.nextcloud_url.rstrip("/")
        self.username = config.nextcloud_username
        self.webdav_url = f"{self.base_url}/remote.php/dav/files/{quote(self.username)}/"
        self.auth = HTTPBasicAuth(config.nextcloud_username, config.nextcloud_password)

    def _remote_folder(self) -> str:
        return self.config.get_remote_folder().strip().strip("/")

    @staticmethod
    def _normalize_name(name: str) -> str:
        return Path(name.strip()).stem.lower()

    def _webdav_file_url(self, *parts: str) -> str:
        encoded_parts = [
            quote(part.strip("/"), safe="")
            for value in parts
            for part in value.split("/")
            if part.strip("/")
        ]
        return self.webdav_url + "/".join(encoded_parts)

    def _resolve_requested_video(
        self,
        requested: str,
        available_videos: Iterable[str],
    ) -> str | None:
        requested_clean = requested.strip()
        if not requested_clean:
            return None

        available_list = list(available_videos)
        available_map_exact = {video.lower(): video for video in available_list}
        exact = available_map_exact.get(requested_clean.lower())
        if exact:
            return exact

        requested_stem = self._normalize_name(requested_clean)
        stem_matches = [
            video
            for video in available_list
            if self._normalize_name(video) == requested_stem
        ]
        if len(stem_matches) == 1:
            return stem_matches[0]
        return None

    def test_connection(self) -> bool:
        remote_folder = self._remote_folder()
        print(f"Probando conexion a Nextcloud: {remote_folder}")

        try:
            response = requests.request(
                method="PROPFIND",
                url=f"{self._webdav_file_url(remote_folder)}/",
                auth=self.auth,
                headers={"Depth": "0"},
                timeout=15,
            )
        except Exception as exc:
            print(f"Error de conexion: {exc}")
            return False

        if response.status_code == 207:
            return True
        if response.status_code == 404:
            print("Carpeta remota no encontrada")
            return False
        if response.status_code == 401:
            print("Credenciales incorrectas")
            return False

        print(f"Error HTTP: {response.status_code}")
        return False

    def get_all_available_videos(self) -> list[str]:
        remote_folder = self._remote_folder()
        print(f"Obteniendo videos desde {remote_folder}")

        try:
            response = requests.request(
                method="PROPFIND",
                url=f"{self._webdav_file_url(remote_folder)}/",
                auth=self.auth,
                headers={"Depth": "1"},
                timeout=15,
            )
        except Exception as exc:
            print(f"Error listando videos: {exc}")
            return []

        if response.status_code != 207:
            print(f"Error obteniendo listado: {response.status_code}")
            return []

        return self._parse_video_listing(response.text)

    def _parse_video_listing(self, xml_text: str) -> list[str]:
        videos: set[str] = set()
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            print(f"Error parseando respuesta WebDAV: {exc}")
            return []

        for href in root.findall(".//{DAV:}href"):
            if not href.text:
                continue
            path = unquote(urlsplit(href.text).path)
            name = Path(path.rstrip("/")).name
            if Path(name).suffix.lower() in self.VIDEO_EXTENSIONS:
                videos.add(name)

        return sorted(videos)

    def validate_specific_videos(
        self,
        specific_videos: list[str] | None = None,
    ) -> list[str]:
        videos_to_validate = specific_videos or self.config.specific_videos
        all_available = self.get_all_available_videos()
        if not all_available:
            print("No se pudo obtener la lista de videos del servidor")
            return []

        valid_videos = []
        invalid_videos = []
        for requested in videos_to_validate:
            resolved = self._resolve_requested_video(requested, all_available)
            if resolved:
                valid_videos.append(resolved)
                suffix = "" if resolved == requested else f" -> {resolved}"
                print(f"OK {requested}{suffix}")
            else:
                invalid_videos.append(requested)
                print(f"No encontrado: {requested}")

        if invalid_videos:
            print(f"No encontrados: {len(invalid_videos)}")

        return valid_videos

    def download_video(
        self,
        video_filename: str,
        videos_folder: Path | str | None = None,
    ) -> Path | None:
        dest_folder = Path(videos_folder) if videos_folder else self.config.videos_folder
        remote_folder = self._remote_folder()
        url = self._webdav_file_url(remote_folder, video_filename)

        print(f"Descargando: {video_filename}")
        try:
            response = requests.get(url, auth=self.auth, stream=True, timeout=60)
            response.raise_for_status()

            dest_folder.mkdir(parents=True, exist_ok=True)
            local_video_path = dest_folder / video_filename
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with local_video_path.open("wb") as f:
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
                            f"\r{progress:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                            end="",
                            flush=True,
                        )

            if total_size > 0:
                print()
            print(f"Guardado en: {local_video_path}")
            return local_video_path
        except Exception as exc:
            print(f"Error descargando {video_filename}: {exc}")
            return None

    def extract_all_frames_from_video(
        self,
        video_filename: str,
        video_path: str | Path,
        quality: int | None = None,
        interval_seconds: float | None = None,
    ) -> tuple[bool, Path, int]:
        video_path = Path(video_path)
        video_name = Path(video_filename).stem
        frames_folder = self.config.frames_root / video_name
        frames_folder.mkdir(parents=True, exist_ok=True)

        jpeg_quality = int(quality or self.config.quality)
        interval = self._parse_runtime_interval(interval_seconds)

        if self.config.use_ffmpeg:
            result = self._extract_with_ffmpeg(video_path, video_name, frames_folder, interval)
            if result[0]:
                return result
            print("ffmpeg fallo, usando OpenCV.")

        return self._extract_with_opencv(
            video_path=video_path,
            video_name=video_name,
            frames_folder=frames_folder,
            jpeg_quality=jpeg_quality,
            interval=interval,
        )

    def _parse_runtime_interval(self, interval_seconds: float | None) -> float:
        try:
            value = (
                interval_seconds
                if interval_seconds is not None
                else self.config.interval_seconds
            )
            return max(float(value), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _extract_with_ffmpeg(
        self,
        video_path: Path,
        video_name: str,
        frames_folder: Path,
        interval: float,
    ) -> tuple[bool, Path, int]:
        out_pattern = frames_folder / f"{video_name}_frame%06d.jpg"
        qscale = int(getattr(self.config, "ffmpeg_qscale", 2))

        cmd_parts = [
            "ffmpeg -y -loglevel error -stats",
            f"-i {shlex.quote(str(video_path))}",
            "-map 0:v:0",
            "-vsync 0",
        ]
        if interval > 0:
            cmd_parts.append(f"-vf fps=1/{interval}")
        cmd_parts.extend(
            [
                "-start_number 0",
                f"-q:v {qscale}",
                shlex.quote(str(out_pattern)),
            ]
        )

        try:
            subprocess.run(" ".join(cmd_parts), shell=True, check=True)
        except Exception as exc:
            print(f"Detalle ffmpeg: {exc}")
            return False, frames_folder, 0

        count = len(list(frames_folder.glob(f"{video_name}_frame*.jpg")))
        print(f"Extraidos {count} frames con ffmpeg -> {frames_folder}")
        return True, frames_folder, count

    def _extract_with_opencv(
        self,
        video_path: Path,
        video_name: str,
        frames_folder: Path,
        jpeg_quality: int,
        interval: float,
    ) -> tuple[bool, Path, int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("No se pudo abrir el video")
            return False, frames_folder, 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_interval = 1
        if interval > 0 and fps > 0:
            frame_interval = max(int(round(fps * interval)), 1)

        frame_index = 0
        saved = 0
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if interval == 0 or frame_index % frame_interval == 0:
                out_path = frames_folder / f"{video_name}_frame{frame_index:06d}.jpg"
                if not cv2.imwrite(str(out_path), frame, encode_param):
                    cap.release()
                    print(f"Error guardando: {out_path}")
                    return False, frames_folder, saved
                saved += 1

                if saved % 500 == 0:
                    total_msg = total if total > 0 else "?"
                    print(f"... {saved} frames guardados (leidos {frame_index + 1}/{total_msg})")

            frame_index += 1

        cap.release()
        print(f"Extraidos {saved} frames con OpenCV -> {frames_folder}")
        return True, frames_folder, saved

    def process_specific_videos(self) -> dict[str, int]:
        valid_videos = self.validate_specific_videos()
        if not valid_videos:
            print("No hay videos validos para procesar")
            return {"success": 0, "total": 0, "frames_extracted": 0}

        print(f"Se procesaran {len(valid_videos)} videos")
        print(f"Intervalo: {self.config.interval_seconds}s")
        print(f"Calidad: {self.config.quality}")
        print(f"ffmpeg: {'si' if self.config.use_ffmpeg else 'no'}")

        answer = input("Continuar? (y/N): ").strip().lower()
        if answer not in {"y", "yes", "s", "si"}:
            print("Procesamiento cancelado")
            return {"success": 0, "total": 0, "frames_extracted": 0}

        success_count = 0
        total_frames_extracted = 0
        for idx, video in enumerate(valid_videos, 1):
            print(f"[{idx}/{len(valid_videos)}] {video}")
            local_video_path = self.download_video(video)
            if not local_video_path:
                continue

            ok, frames_folder, n_frames = self.extract_all_frames_from_video(
                video,
                local_video_path,
            )
            if ok:
                success_count += 1
                total_frames_extracted += n_frames
                print(f"Frames en: {frames_folder}")

        return {
            "success": success_count,
            "total": len(valid_videos),
            "frames_extracted": total_frames_extracted,
        }
