# Pipeline Desde Videos Locales

Este flujo toma videos desde una carpeta local de entrada. No descarga videos desde Nextcloud.

```text
videos locales
-> extraccion de frames
-> YOLO-seg a mascaras RGB compatibles
-> analitico geometrico
-> resultados
```

## Requisitos

El contenedor de percepcion debe estar construido y levantado:

```powershell
cd D:\Memoria\Cono\perception
docker compose up -d
```

Si todavia no existe la imagen:

```powershell
docker compose build
docker compose up -d
```

## Carpeta De Entrada

Coloca los videos en una carpeta local, por ejemplo:

```text
D:\Memoria\Cono\data\pipeline_input\videos\
```

Extensiones soportadas:

```text
.mp4, .avi, .mov, .mkv, .m4v
```

## Ejecutar

Desde la raiz del proyecto:

```powershell
cd D:\Memoria\Cono

python scripts/run_pipeline_from_videos.py `
  --videos-dir "data\pipeline_input\videos" `
  --run-name "prueba_videos_01" `
  --yolo-model "perception\models\reference\carga_visible_yolov8n-seg_20260527_195509\best.pt" `
  --interval-seconds 2 `
  --conf 0.35 `
  --device 0 `
  --overwrite
```

Si quieres que el script levante Docker antes de inferir:

```powershell
python scripts/run_pipeline_from_videos.py `
  --videos-dir "data\pipeline_input\videos" `
  --run-name "prueba_videos_01" `
  --start-docker `
  --overwrite
```

## Salidas

Frames y mascaras, en una carpeta visible para Docker:

```text
D:\Memoria\Cono\perception\model_testing\pipeline_runs\prueba_videos_01\
  frames\
  yolo_masks\
```

Resultados finales:

```text
D:\Memoria\Cono\outputs\pipeline_runs\prueba_videos_01\
  analytic\
    tables\features.csv
    overlays\
  manifests\
    videos.csv
    frames.csv
    yolo_manifest.csv
    yolo_summary.json
  pipeline_summary.json
```

## Parametros Utiles

- `--interval-seconds 2`: extrae un frame cada 2 segundos.
- `--interval-seconds 0`: extrae todos los frames.
- `--max-frames-per-video 20`: limita frames por video para una prueba rapida.
- `--limit-videos 1`: procesa solo el primer video encontrado.
- `--recursive`: busca videos en subcarpetas.
- `--mask-mode largest`: conserva la instancia principal detectada por YOLO.
- `--mask-mode merge_all`: une todas las instancias detectadas por YOLO.

## Nota

El script usa Docker solo para la etapa YOLO. La extraccion de frames y el analitico corren con el Python local del proyecto.
