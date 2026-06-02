# Caso de referencia: prueba_01

Este snapshot conserva una prueba de inferencia ejecutada con:

```text
models/reference/carga_visible_yolov8n-seg_20260527_195509/best.pt
```

Contenido:

```text
sources/   24 imagenes originales
outputs/   24 imagenes con las predicciones dibujadas por YOLO-seg
```

El caso sirve como referencia visual inicial. Las pruebas futuras deben ejecutarse en `model_testing/sources/` y `model_testing/outputs/`, que permanecen fuera de Git.

Para repetir la inferencia sobre las entradas versionadas:

```powershell
docker compose exec cone-yolo python scripts/predict_yolo_seg.py `
  --model models/reference/carga_visible_yolov8n-seg_20260527_195509/best.pt `
  --source model_testing/reference/prueba_01/sources `
  --project model_testing/outputs `
  --name prueba_01_repetida `
  --conf 0.35 `
  --device 0
```
