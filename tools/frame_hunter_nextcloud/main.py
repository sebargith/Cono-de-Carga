#!/usr/bin/env python3
"""
Frame Hunter - Nextcloud + extracción de frames
==============================================
Versión mínima del proyecto original, enfocada solo en:
1) conexión/listado de videos en Nextcloud
2) descarga de videos
3) extracción de frames
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.config import load_config
from src.nextcloud_manager import NextCloudManager


def print_banner():
    print()
    print("🎯" * 26)
    print("🎯  FRAME HUNTER - NEXTCLOUD/FRAMES  🎯")
    print("🎯" * 26)
    print()


def print_menu():
    print("1) Probar conexión")
    print("2) Ver todos los videos disponibles")
    print("3) Validar videos específicos configurados")
    print("4) Descargar + extraer videos específicos")
    print("5) Salir")
    print()


def main():
    print_banner()

    try:
        config = load_config()
        manager = NextCloudManager(config)
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        return 1

    while True:
        print(f"📂 Carpeta remota: {config.get_remote_folder()}")
        print(f"💾 Videos locales: {config.videos_folder}")
        print(f"🖼️  Frames locales: {config.frames_root}")
        print(f"⏱️  Intervalo: {config.interval_seconds}s (0 = todos los frames)")
        print()
        print_menu()

        try:
            choice = input("Selecciona una opción: ").strip()
        except KeyboardInterrupt:
            print("\n❌ Cancelado")
            return 1

        if choice == "1":
            ok = manager.test_connection()
            print("✅ Conexión OK" if ok else "❌ Conexión fallida")

        elif choice == "2":
            videos = manager.get_all_available_videos()
            if videos:
                print(f"\n📹 Videos encontrados: {len(videos)}")
                for i, video in enumerate(videos, 1):
                    print(f"{i:3d}. {video}")
            else:
                print("❌ No se encontraron videos")

        elif choice == "3":
            videos = manager.validate_specific_videos()
            print(f"\n✅ Válidos: {len(videos)}")

        elif choice == "4":
            stats = manager.process_specific_videos()
            print("\n📊 Resumen final")
            print(f"   ✅ Exitosos: {stats['success']}")
            print(f"   📹 Intentados: {stats['total']}")
            print(f"   🖼️  Frames guardados: {stats['frames_extracted']}")

        elif choice == "5":
            print("👋 Saliendo")
            return 0

        else:
            print("❌ Opción inválida")

        print()
        input("Presiona Enter para continuar...")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
