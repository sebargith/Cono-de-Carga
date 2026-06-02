# Modelo YOLO-seg de referencia

Este directorio conserva un unico modelo entrenado como referencia inicial del proyecto:

```text
carga_visible_yolov8n-seg_20260527_195509/
  best.pt
  args.yaml
```

`best.pt` corresponde al mejor checkpoint de la corrida indicada. `args.yaml` conserva los argumentos efectivos usados por Ultralytics.

Los modelos generados por futuros experimentos deben permanecer fuera de Git y registrarse mediante MLflow. Solo se debe agregar otro checkpoint al repositorio si se decide reemplazar explicitamente esta referencia.
