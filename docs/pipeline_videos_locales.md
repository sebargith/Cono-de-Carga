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
  statistics\
    indicator_summary.csv
    indicator_distributions.csv
  video_summary.csv
  pipeline_summary.json
```

## Veredicto Por Video

El pipeline genera un resumen por video en:

```text
D:\Memoria\Cono\outputs\pipeline_runs\<run_name>\video_summary.csv
```

Cada fila resume los frames asociados a un video:

```text
video_name
total_frames
valid_frames
valid_ratio
centrado_ratio
izquierda_ratio
derecha_ratio
p25_peak_x_norm
median_peak_x_norm
p75_peak_x_norm
p25_area_symmetry
median_area_symmetry
p75_area_symmetry
p25_centroid_x_norm
median_centroid_x_norm
p75_centroid_x_norm
final_verdict
confidence
verdict_reason
```

`statistics\indicator_summary.csv` entrega cantidad, P25, mediana y P75 de
`peak_x_norm`, `area_symmetry` y `centroid_x_norm`, tanto para la corrida
completa como para cada video. `statistics\indicator_distributions.csv`
entrega sus histogramas tabulados en 20 intervalos. Solo se consideran frames
validos.

`pipeline_summary.json` registra tambien:

- los umbrales efectivos de pico y simetria usados por el analitico;
- la inferencia YOLO media por frame y acumulada, reportada por Ultralytics;
- el tiempo completo empleado para generar las mascaras analiticas.

Veredictos posibles:

```text
carga_centrada
tendencia_izquierda
tendencia_derecha
mixto_o_inestable
no_evaluable
```

Regla inicial:

- si hay pocos frames evaluables, el video queda `no_evaluable`;
- si una clase domina al menos el umbral configurado, se usa esa tendencia;
- si no hay mayoria clara, se usan las medianas de `peak_x_norm` y `area_symmetry`;
- si aun asi no hay una tendencia clara, queda `mixto_o_inestable`.

## Parametros Utiles

- `--interval-seconds 2`: extrae un frame cada 2 segundos.
- `--interval-seconds 0`: extrae todos los frames.
- `--max-frames-per-video 20`: limita frames por video para una prueba rapida.
- `--limit-videos 1`: procesa solo el primer video encontrado.
- `--recursive`: busca videos en subcarpetas.
- `--mask-mode largest`: conserva la instancia principal detectada por YOLO.
- `--mask-mode merge_all`: une todas las instancias detectadas por YOLO.
- `--video-min-valid-ratio 0.5`: minima fraccion de frames validos para emitir veredicto.
- `--video-majority-ratio 0.6`: fraccion minima de frames validos que deben apoyar una clase dominante.

## Nota

El script usa Docker solo para la etapa YOLO. La extraccion de frames y el analitico corren con el Python local del proyecto.
