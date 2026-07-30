# Frame Hunter - Nextcloud + Frames

Versión mínima del repo, recortada para hacer solo esto:

- conectarse a Nextcloud
- listar videos de una carpeta remota específica
- validar una lista de videos específicos
- descargar videos
- extraer frames

## Configuración

Ahora **no** usa `camera.name`, `subfolder` ni rutas armadas por partes.

La carpeta remota y la lista de videos se configuran en `config/config.yaml`:

```yaml
nextcloud:
  remote_folder: "Data-Analitica/ChancadorCMDIC"
```

Las credenciales se mantienen fuera de Git. Crea `.env` a partir de
`.env.example`:

```powershell
cd D:\Memoria\Cono\tools\frame_hunter_nextcloud
Copy-Item .env.example .env
```

Y la lista de videos se busca **solo dentro de esa carpeta**:

```yaml
specific_videos:
  - "Camera_8_20241026_164010"
  - "Camera_8_20241026_170010.mp4"
```

Acepta nombres **con o sin extensión**.

## Qué se eliminó

- inferencia YOLO
- previews MP4
- dependencias de `ultralytics`
- partes del flujo original que estaban acopladas a inferencia

## Estructura

- `main.py`: menú simple para usar la herramienta
- `src/config.py`: carga de configuración
- `src/nextcloud_manager.py`: Nextcloud + descarga + extracción de frames
- `config/config.yaml`: configuración principal

## Uso

```powershell
cd D:\Memoria\Cono\tools\frame_hunter_nextcloud
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Antes de ejecutar, completa `.env` y revisa `nextcloud.remote_folder`.

## Notas

- Si `interval_seconds: 0`, extrae todos los frames.
- Si `interval_seconds: 5`, extrae 1 frame cada 5 segundos.
- Si `use_ffmpeg: true`, intenta usar `ffmpeg` primero. Si falla, usa OpenCV.
- Los videos descargados quedan en `data/videos/`
- Los frames quedan en `data/frames/<nombre_video>/`
