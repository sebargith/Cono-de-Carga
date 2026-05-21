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
- `corrido_izquierda`
- `corrido_derecha`
- `no_evaluable`

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
- Entrenar un modelo de percepción para segmentar automáticamente la carga visible.
- Conectar la máscara predicha por el modelo con el mismo analítico geométrico.
- Comparar resultados usando máscaras manuales vs máscaras automáticas.
- Evaluar resultados por secuencia de video, no solo por frame.


## Uso Básico

```text
python Memoria/Cono/scripts/run_simple_analytic.py \
  --images-dir <ruta_a_JPEGImages> \
  --masks-dir <ruta_a_SegmentationClass> \
  --output-dir Memoria/Cono/outputs \
  --config Memoria/Cono/configs/analytic_config.yaml
```
