# Resumen del pipeline y la web app

Este documento resume como funciona el pipeline completo del proyecto Cono de Carga y como se usa la web app que lo ejecuta.

## 1. Pipeline

El pipeline une tres bloques que ya existian por separado:

1. extraccion de frames desde videos,
2. segmentacion automatica de la carga visible con YOLO,
3. analitico geometrico para clasificar la forma visible de la carga.

El script principal es:

```text
scripts/run_pipeline_from_videos.py
```

Este script actua como orquestador. No reemplaza al extractor, al modelo YOLO ni al analitico; solo ejecuta cada etapa en orden y mueve los archivos de salida de una etapa hacia la entrada de la siguiente.

### Flujo resumido

```text
videos
  -> frames JPG
  -> mascaras generadas por YOLO
  -> features geometricas por frame
  -> veredicto por video
  -> overlays, tablas y resumen
```

### Entradas principales

El pipeline recibe una carpeta con videos:

```text
--videos-dir <carpeta_con_videos>
```

Tambien recibe parametros como:

```text
--interval-seconds
--max-frames-per-video
--conf
--device
--run-name
```

### Salidas principales

Para cada corrida se generan dos tipos de carpetas:

```text
work/
  frames/
  yolo_masks/

results/
  analytic/
    overlays/
    tables/features.csv
  manifests/
  statistics/
    indicator_summary.csv
    indicator_distributions.csv
  video_summary.csv
  pipeline_summary.json
```

`work/` contiene archivos intermedios.

`results/` contiene los resultados que se revisan o descargan.

### Como se conectan las etapas

La conexion entre etapas se hace por nombres de archivo y carpetas.

El extractor genera frames:

```text
work/frames/video__frame000000.jpg
```

YOLO genera mascaras con el mismo nombre base:

```text
work/yolo_masks/video__frame000000.png
```

El analitico recibe ambas carpetas:

```text
images-dir = work/frames
masks-dir  = work/yolo_masks
```

Luego genera:

```text
results/analytic/tables/features.csv
results/analytic/overlays/
```

Finalmente, el pipeline agrupa los resultados por video usando:

```text
results/manifests/frames.csv
results/manifests/videos.csv
results/analytic/tables/features.csv
```

Con eso produce:

```text
results/video_summary.csv
```

Ese archivo contiene el veredicto final por video.

Ademas, `results/statistics/` contiene resultados descriptivos para los frames
validos de la corrida y de cada video:

- `indicator_summary.csv`: cantidad, P25, mediana y P75 de `peak_x_norm`,
  `area_symmetry` y `centroid_x_norm`;
- `indicator_distributions.csv`: histogramas tabulados de 20 intervalos para
  esos mismos indicadores.

Estas variables describen la distribucion visual interna de la carga
segmentada. Sus signos indican tendencia hacia izquierda o derecha respecto de
la propia mascara, no balance fisico ni distribucion real del payload.

## 2. Web app

La web app esta en:

```text
web_app/
```

Su objetivo es dar una interfaz simple para ejecutar el pipeline sin escribir el comando completo.

La web app permite:

1. subir uno o mas videos,
2. configurar parametros basicos,
3. ejecutar el pipeline,
4. ver la consola en vivo,
5. revisar overlays,
6. ver el veredicto,
7. revisar estadisticas por frame,
8. descargar resultados.

### Que hace internamente

Cuando se presiona `Ejecutar pipeline`, la web app:

1. guarda los videos en:

```text
web_app/data/uploads/<run_name>/videos/
```

2. crea carpetas de trabajo y resultados:

```text
web_app/outputs/runs/<run_name>/work/
web_app/outputs/runs/<run_name>/results/
```

3. arma el comando para ejecutar:

```text
scripts/run_pipeline_from_videos.py
```

4. muestra la salida del proceso en una consola en vivo.

5. al terminar, lee los archivos generados y los muestra en pantalla.

La web app llama al pipeline desde:

```text
web_app/src/pipeline_runner.py
```

Ese archivo construye el comando y lo ejecuta con salida en streaming.

### Vista de resultados

Despues de correr una prueba, la web app muestra:

1. titulo del video,
2. overlays,
3. veredicto,
4. estadisticas,
5. resultados por frame,
6. botones de descarga.

Si la corrida tiene mas de un video, la app permite seleccionar el video activo y filtra overlays, features y veredicto para ese video.

## 3. Herramientas y tecnologias utilizadas

### Python

Se usa como lenguaje principal del proyecto.

Uso en el pipeline:

- ejecutar scripts,
- leer y escribir CSV/JSON,
- controlar el flujo entre etapas,
- llamar procesos externos como Docker.

### OpenCV

Se usa para manejar videos e imagenes.

Uso en el pipeline:

- abrir videos,
- leer frames,
- guardar frames como JPG,
- procesar mascaras e imagenes dentro del analitico.

### YOLO / Ultralytics

Se usa para segmentar automaticamente la carga visible.

Uso en el pipeline:

- recibe frames,
- predice la mascara de carga,
- guarda mascaras compatibles con el analitico.

### Docker

Se usa para encapsular el entorno YOLO.

Uso en el pipeline:

- ejecutar Ultralytics/YOLO con sus dependencias,
- usar GPU si esta disponible,
- evitar instalar PyTorch y CUDA directamente en la venv local de la web app.

El pipeline llama YOLO con:

```text
docker compose exec -T cone-yolo ...
```

Por eso, para el pipeline completo, Docker debe estar levantado.

### NumPy

Se usa para operaciones numericas sobre mascaras, perfiles y features.

Uso en el analitico:

- calcular areas,
- perfiles superiores,
- normalizaciones,
- mediciones de simetria e irregularidad.

### SciPy

Se usa como apoyo para suavizado y deteccion de peaks cuando corresponde.

Uso en el analitico:

- suavizar perfiles,
- detectar peaks del perfil superior.

### Pandas

Se usa para manejar tablas.

Uso en la web app:

- leer `features.csv`,
- leer `video_summary.csv`,
- mostrar resultados por frame,
- filtrar resultados por video.

### Streamlit

Se usa para construir la web app local.

Uso en la web app:

- formulario de carga de videos,
- controles de parametros,
- consola en vivo,
- visualizacion de overlays,
- tablas,
- metricas,
- botones de descarga.

### YAML

Se usa para configurar umbrales y parametros del analitico.

Archivo principal:

```text
configs/analytic_config.yaml
```

Uso en el pipeline:

- el analitico lee esta configuracion para clasificar frames.

### CSV y JSON

Se usan como formato de intercambio simple entre etapas.

CSV:

- `frames.csv`,
- `videos.csv`,
- `features.csv`,
- `video_summary.csv`.

JSON:

- `pipeline_summary.json`,
- `yolo_summary.json`.

Estos archivos hacen que cada etapa sea revisable y facil de depurar.

## 4. Idea principal

El pipeline funciona porque cada etapa tiene una entrada y una salida clara:

```text
extractor: video -> frames
YOLO: frames -> mascaras
analitico: frames + mascaras -> features + overlays
resumen: features -> veredicto por video
web app: archivos de resultado -> visualizacion
```

La web app solo entrega una forma comoda de operar este flujo y revisar los resultados. La logica principal sigue estando en los scripts del pipeline, el modelo YOLO y el analitico geometrico.
