# Modelos de segmentación

El pipeline considera un único modelo entrenado de segmentación:

- `primary/psinet_carga_visible.pt`: modelo usado por el flujo vigente. La carga
  corresponde a la clase `material` (`class_id=1`).

`alternative/best.pt` es un checkpoint histórico del proceso de desarrollo. Se
mantiene únicamente como antecedente reproducible del entrenamiento y no se
considera un segundo modelo del proyecto ni parte del flujo vigente.

La selección predeterminada está declarada en:

- `perception/configs/perception_config.yaml`
- `scripts/run_pipeline_from_videos.py`

Los nombres de las carpetas `primary/` y `alternative/` se conservan por
compatibilidad con las rutas existentes, no para definir una comparación de
modelos.

Al usar el modelo vigente no es necesario pasar `--class-id`: la configuración
selecciona automáticamente la clase 1.
