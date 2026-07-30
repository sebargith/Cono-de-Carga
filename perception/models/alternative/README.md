# Modelo alternativo

`best.pt` corresponde al checkpoint YOLOv8n-seg entrenado durante el desarrollo
del proyecto. Se conserva para comparación, respaldo y pruebas controladas.

Para utilizarlo en el pipeline:

```powershell
python scripts/run_pipeline_from_videos.py `
  --videos-dir "data\pipeline_input\videos" `
  --yolo-model "perception\models\alternative\best.pt" `
  --class-id 0 `
  --run-name "prueba_modelo_alternativo" `
  --overwrite
```

Este modelo contiene una sola clase de carga, con `class_id=0`.
