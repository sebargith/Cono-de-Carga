#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.nextcloud_manager import NextCloudManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga videos desde Nextcloud y extrae frames."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Ruta al archivo config.yaml. Por defecto usa config/config.yaml.",
    )
    return parser.parse_args()


def print_banner() -> None:
    print()
    print("=" * 50)
    print("FRAME HUNTER - NEXTCLOUD/FRAMES")
    print("=" * 50)
    print()


def print_menu() -> None:
    print("1) Probar conexion")
    print("2) Ver todos los videos disponibles")
    print("3) Validar videos especificos configurados")
    print("4) Descargar + extraer videos especificos")
    print("5) Salir")
    print()


def main() -> int:
    args = parse_args()
    print_banner()

    try:
        config = load_config(args.config)
        manager = NextCloudManager(config)
    except Exception as exc:
        print(f"Error cargando configuracion: {exc}")
        return 1

    while True:
        print(f"Carpeta remota: {config.get_remote_folder()}")
        print(f"Videos locales: {config.videos_folder}")
        print(f"Frames locales: {config.frames_root}")
        print(f"Intervalo: {config.interval_seconds}s (0 = todos los frames)")
        print()
        print_menu()

        try:
            choice = input("Selecciona una opcion: ").strip()
        except KeyboardInterrupt:
            print("\nCancelado")
            return 1

        if choice == "1":
            ok = manager.test_connection()
            print("Conexion OK" if ok else "Conexion fallida")
        elif choice == "2":
            videos = manager.get_all_available_videos()
            if videos:
                print(f"\nVideos encontrados: {len(videos)}")
                for i, video in enumerate(videos, 1):
                    print(f"{i:3d}. {video}")
            else:
                print("No se encontraron videos")
        elif choice == "3":
            videos = manager.validate_specific_videos()
            print(f"\nValidos: {len(videos)}")
        elif choice == "4":
            stats = manager.process_specific_videos()
            print("\nResumen final")
            print(f"Exitosos: {stats['success']}")
            print(f"Intentados: {stats['total']}")
            print(f"Frames guardados: {stats['frames_extracted']}")
        elif choice == "5":
            print("Saliendo")
            return 0
        else:
            print("Opcion invalida")

        print()
        input("Presiona Enter para continuar...")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
