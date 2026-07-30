# Caso de referencia: prueba_01

Este snapshot conserva una prueba de inferencia ejecutada con:

```text
models/alternative/best.pt
```

Contenido:

```text
sources/   24 imagenes originales
outputs/   24 imagenes con las predicciones dibujadas por YOLO-seg
```

El caso sirve como referencia visual inicial. Las pruebas futuras deben ejecutarse en `model_testing/sources/` y `model_testing/outputs/`, que permanecen fuera de Git.

Para repetir la inferencia sobre las entradas versionadas:

```powershell
cd D:\Memoria\Cono
.\web_app\.venv\Scripts\Activate.ps1

python perception\scripts\predict_yolo_seg.py `
  --model perception\models\alternative\best.pt `
  --source perception\model_testing\reference\prueba_01\sources `
  --project perception\model_testing\outputs `
  --name prueba_01_repetida `
  --conf 0.35 `
  --device 0
```
