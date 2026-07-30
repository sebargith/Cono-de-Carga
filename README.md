# Cono de Carga en CAEX

Sistema de visión computacional para analizar la **forma visible de la carga**
en camiones CAEX observados desde atrás. El flujo actual procesa videos locales,
segmenta automáticamente la carga y clasifica cada frame como `izquierda`,
`centrado`, `derecha` o `no_evaluable`.

Las mediciones son internas a la máscara segmentada. Describen la posición
relativa del pico, la simetría interna de área y la posición relativa del
centroide; no representan balance físico, centro de masa real ni distribución
real del payload.

## Pipeline

```text
video local
→ extracción de frames
→ segmentación de carga con YOLO
→ máscara RGB compatible con el analítico
→ analítico geométrico por frame
→ veredicto y estadísticas por video
→ CSV, JSON y overlays
```

El modelo principal es `perception/models/primary/psinet_carga_visible.pt`. La
carga corresponde a la clase `material` (`class_id=1`). Como alternativa se
conserva `perception/models/alternative/best.pt`, entrenado en este proyecto y
con clase `Carga` (`class_id=0`).

## Ejecución desde video

La inferencia de percepción se ejecuta dentro del contenedor Docker:

```powershell
cd D:\Memoria\Cono\perception
docker compose up -d

cd D:\Memoria\Cono
python scripts\run_pipeline_from_videos.py `
  --videos-dir "data\pipeline_input\videos" `
  --run-name "prueba_video" `
  --interval-seconds 0.5 `
  --device 0 `
  --overwrite
```

Los resultados quedan en `outputs/pipeline_runs/<run_name>/`. La guía completa
está en [docs/pipeline_videos_locales.md](docs/pipeline_videos_locales.md).

## Aplicación web

En Windows se puede iniciar con doble clic en
`Abrir web app Cono de Carga.bat`. El lanzador crea el entorno local de la
interfaz cuando no existe. Docker debe estar disponible para la etapa de
segmentación.

## Analítico con máscaras existentes

También es posible ejecutar solo el analítico con máscaras manuales de CVAT o
máscaras ya generadas:

```powershell
python scripts\run_simple_analytic.py `
  --images-dir "<ruta_a_JPEGImages>" `
  --masks-dir "<ruta_a_SegmentationClass>" `
  --output-dir "outputs\analitico_manual" `
  --config "configs\analytic_config.yaml"
```

## Estructura

```text
configs/       configuración del analítico
data/          datos locales y benchmarks no versionados
deliverables/  informe, presentación y evaluaciones consolidadas
docs/          arquitectura, uso y contexto técnico
perception/    preparación, entrenamiento e inferencia YOLO con Docker
scripts/       orquestadores y evaluaciones
src/           analítico geométrico
tools/         utilidades auxiliares y generación de figuras
web_app/       interfaz Streamlit encapsulada
```

La documentación de entrenamiento está en
[perception/README.md](perception/README.md), y la organización de modelos en
[perception/models/README.md](perception/models/README.md).
