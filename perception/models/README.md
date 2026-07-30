# Modelos de segmentación

El pipeline utiliza un modelo principal y conserva un checkpoint alternativo.

- `primary/psinet_carga_visible.pt`: modelo PSINet usado por defecto. La carga
  corresponde a la clase `material` (`class_id=1`).
- `alternative/best.pt`: checkpoint YOLOv8n-seg entrenado en este proyecto. Su
  clase `Carga` corresponde a `class_id=0`.

La selección predeterminada está declarada en:

- `perception/configs/perception_config.yaml`
- `scripts/run_pipeline_from_videos.py`

Ambos archivos `.pt` deben permanecer dentro de `perception/models/` para que
sean visibles desde el contenedor Docker.

Al usar el modelo principal no es necesario pasar `--class-id`: la
configuración selecciona automáticamente la clase 1. Para ejecutar el modelo
alternativo se debe indicar `--class-id 0`.
