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

El pipeline considera un único modelo entrenado de segmentación:
`perception/models/primary/psinet_carga_visible.pt`. La carga corresponde a la
clase `material` (`class_id=1`). La segmentación es una etapa auxiliar; el
resultado del proyecto corresponde al analítico geométrico aplicado sobre sus
máscaras.

## Ejecución desde video

La inferencia se ejecuta localmente con el mismo entorno Python del pipeline.
El lanzador de la web app crea este entorno e instala las dependencias cuando
faltan. Para prepararlo manualmente:

```powershell
cd D:\Memoria\Cono
py -3.12 -m venv web_app\.venv
.\web_app\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\runtime.txt

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
interfaz e instala Streamlit, PyTorch y Ultralytics cuando faltan. La primera
instalación puede tardar varios minutos; los inicios siguientes reutilizan el
mismo entorno. No es necesario levantar Docker para ejecutar la web app o el
pipeline.

## Uso de Docker

Docker se conserva en `perception/` para preparar datasets, entrenar y validar
modelos con un entorno reproducible de CUDA, PyTorch, Ultralytics y MLflow. Los
pesos producidos se copian a `perception/models/` y la inferencia operacional
los carga localmente.

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
perception/    modelos, inferencia local y entrenamiento YOLO con Docker
requirements/  dependencias del pipeline local
scripts/       orquestadores y evaluaciones
src/           analítico geométrico
tools/         utilidades auxiliares y generación de figuras
web_app/       interfaz Streamlit encapsulada
```

La documentación de entrenamiento está en
[perception/README.md](perception/README.md), y la organización de modelos en
[perception/models/README.md](perception/models/README.md).
