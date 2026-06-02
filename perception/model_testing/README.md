# Pruebas de modelos

Usa este directorio para probar modelos entrenados sin mezclar archivos con el dataset ni con las corridas de entrenamiento.

Estructura:

```text
model_testing/
  models/     pesos a probar, por ejemplo best.pt
  sources/    imagenes, carpetas o videos de entrada
  outputs/    resultados generados por predict_yolo_seg.py
```

Ejemplo desde PowerShell:

```powershell
cd D:\Memoria\Cono\perception

docker compose exec cone-yolo python scripts/predict_yolo_seg.py `
  --model model_testing/models/best.pt `
  --source model_testing/sources `
  --project model_testing/outputs `
  --name prueba_01 `
  --conf 0.35 `
  --device 0 `
  --exist-ok
```

Los resultados quedan en:

```text
D:\Memoria\Cono\perception\model_testing\outputs\prueba_01\
```

El snapshot versionado de la primera prueba se encuentra en:

```text
D:\Memoria\Cono\perception\model_testing\reference\prueba_01\
```
