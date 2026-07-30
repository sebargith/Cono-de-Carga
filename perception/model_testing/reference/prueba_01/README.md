# Caso histórico de referencia: prueba_01

Este snapshot conserva una prueba histórica de inferencia ejecutada con:

```text
models/alternative/best.pt
```

Contenido:

```text
sources/   24 imagenes originales
outputs/   24 imagenes con las predicciones dibujadas por YOLO-seg
```

El caso sirve únicamente como referencia visual inicial y no forma parte del
flujo ni de la evaluación de rendimiento vigentes. Las pruebas futuras deben
ejecutarse en `model_testing/sources/` y `model_testing/outputs/`, que
permanecen fuera de Git.

El comando siguiente se conserva solo para reproducir ese snapshot histórico:

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
