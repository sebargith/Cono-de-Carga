# Frame Hunter - Nextcloud

Herramienta auxiliar para descargar videos desde una carpeta de Nextcloud y extraer frames para el proyecto Cono de Carga.

El flujo cubre solo adquisicion de datos:

```text
Nextcloud
-> listado de videos disponibles
-> validacion de videos especificos
-> descarga local
-> extraccion de frames
```

No ejecuta entrenamiento, inferencia YOLO ni el analitico geometrico.

## Estructura

```text
frame-hunter-nextcloud/
  config/
    config.example.yaml
  src/
    __init__.py
    config.py
    nextcloud_manager.py
  .env.example
  .gitignore
  main.py
  requirements.txt
```

Los videos y frames generados quedan fuera de git:

```text
data/videos/
data/frames/
```

## Instalacion local

Desde este directorio:

```powershell
cd D:\Memoria\Cono\frame-hunter-nextcloud
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si `use_ffmpeg: true`, instala `ffmpeg` y dejalo disponible en el `PATH`. Si `ffmpeg` falla, el script intenta extraer frames con OpenCV.

## Configuracion

Copia los ejemplos:

```powershell
Copy-Item .env.example .env
Copy-Item config\config.example.yaml config\config.yaml
```

Edita `.env` con tus credenciales:

```text
NEXTCLOUD_URL=https://tu-nextcloud
NEXTCLOUD_USERNAME=tu_usuario
NEXTCLOUD_PASSWORD=tu_password
```

Edita `config/config.yaml` para indicar la carpeta remota, videos y parametros de extraccion:

```yaml
nextcloud:
  remote_folder: "Data-Analitica/ChancadorCMDIC"

extraction:
  interval_seconds: 2
  quality: 85
  use_ffmpeg: true
  ffmpeg_qscale: 2

specific_videos:
  - "Camera_8_20240914_142001"
  - "Camera_8_20240910_103001.mp4"
```

`specific_videos` acepta nombres con o sin extension si existe una unica coincidencia.

## Uso

Ejecuta el menu interactivo:

```powershell
python main.py
```

Tambien puedes pasar una configuracion explicita:

```powershell
python main.py --config config\config.yaml
```

Opciones del menu:

- `1`: prueba conexion a Nextcloud.
- `2`: lista videos disponibles en `nextcloud.remote_folder`.
- `3`: valida los videos definidos en `specific_videos`.
- `4`: descarga y extrae frames de los videos validos.
- `5`: salir.

## Salidas

Por defecto:

```text
data/videos/<video>.mp4
data/frames/<nombre_video>/<nombre_video>_frame000000.jpg
```

Si `interval_seconds: 0`, extrae todos los frames. Si `interval_seconds: 2`, extrae un frame cada 2 segundos.

## Archivos que no se deben versionar

Este modulo ignora credenciales, entorno local y datos generados:

```text
.env
config/config.yaml
venv/
.venv/
data/videos/
data/frames/
__pycache__/
```
