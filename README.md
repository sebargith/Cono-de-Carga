# Cono de Carga en CAEX

Este proyecto corresponde al desarrollo inicial de un analítico geométrico para el estudio del **cono de carga en camiones de extracción CAEX**, usando visión computacional aplicada a minería.

El objetivo es analizar la **forma visible de la carga** en la tolva del camión a partir de frames extraídos desde videos con vista posterior. En esta primera etapa se trabaja con máscaras manuales de la carga etiquetadas en CVAT, sin usar todavía una ROI, máscara de tolva ni modelo de percepción automático.

## Estado actual

Actualmente se está desarrollando un analítico simple que toma como entrada:

- imagen original;
- máscara manual de carga exportada desde CVAT.

A partir de esa máscara, el sistema calcula características geométricas como:

- bounding box de la carga;
- centroide;
- perfil superior;
- pico principal;
- simetría interna izquierda/derecha;
- área izquierda y derecha;
- clase preliminar del estado de la carga.

La salida principal clasifica cada frame como:

- `centrado`
- `izquierda`
- `derecha`
- `no_evaluable`

Los frames donde la máscara de carga toca cualquier borde de la imagen se descartan del análisis geométrico. Estos casos se conservan en el CSV para mantener trazabilidad, pero quedan marcados como:

```text
predicted_class = no_evaluable
is_valid = False
error_reason = mascara_toca_borde_imagen
```

El CSV también indica qué bordes fueron alcanzados:

```text
touches_left_border
touches_right_border
touches_top_border
touches_bottom_border
is_truncated_by_frame
```

El margen de descarte se configura en `configs/analytic_config.yaml`:

```yaml
geometry:
  border_touch_margin_px: 0
```

## Flujo actual

```text
frames de video
→ etiquetado manual en CVAT
→ exportación de máscaras
→ lectura de imagen y máscara
→ conversión de máscara a binaria
→ extracción de features geométricas
→ clasificación preliminar
→ generación de CSV y overlays
```

## Pasos futuros

- Validar el analítico con más frames etiquetados manualmente.
- Ajustar los umbrales de clasificación en `analytic_config.yaml`.
- Mejorar la detección de casos `no_evaluable`.
- Agregar evaluación cuantitativa con etiquetas manuales por frame.
- Conectar la máscara predicha por el modelo con el mismo analítico geométrico.
- Comparar resultados usando máscaras manuales vs máscaras automáticas.
- Evaluar resultados por secuencia de video, no solo por frame.

## Modelo de percepción YOLO-seg

El repositorio incluye un módulo independiente en [`perception/`](perception/) para entrenar, validar y probar un modelo YOLO de segmentación de carga visible.

El flujo implementado es:

```text
exportaciones CVAT
→ conversión a dataset YOLO-seg
→ entrenamiento con GPU mediante Docker
→ seguimiento experimental en MLflow
→ validación e inferencia
→ máscara automática para conectar con el analítico geométrico
```

Las instrucciones detalladas se encuentran en [`perception/README.md`](perception/README.md).


## Uso Básico

```text
python Memoria/Cono/scripts/run_simple_analytic.py \
  --images-dir <ruta_a_JPEGImages> \
  --masks-dir <ruta_a_SegmentationClass> \
  --output-dir Memoria/Cono/outputs \
  --config Memoria/Cono/configs/analytic_config.yaml
```

## Pipeline Desde Videos Locales

Para procesar videos ya descargados en una carpeta local, usar:

```text
scripts/run_pipeline_from_videos.py
```

Este flujo extrae frames, genera mascaras automaticas con YOLO-seg y ejecuta el analitico geometrico. La guia esta en [`docs/pipeline_videos_locales.md`](docs/pipeline_videos_locales.md).

Cada corrida tambien genera P25, mediana, P75 e histogramas tabulados de
`peak_x_norm`, `area_symmetry` y `centroid_x_norm` para describir la
distribucion visual interna de la carga.
