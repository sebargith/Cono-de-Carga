# Evaluación del analítico de Cono

La evaluación considera 90 imágenes: 30 referencias de centrado, izquierda y derecha. Las etiquetas se revisaron por el autor junto al equipo de ingenieros de PSINet. La sensibilidad se ajustó en distintas pruebas; este conjunto reporta el rendimiento con la configuración seleccionada.

Se obtuvieron 80/90 aciertos (88,89 %) y 90/90 imágenes evaluables. Los aciertos por clase fueron 22/30, 28/30 y 30/30. Son recall por clase, aunque el JSON histórico los nombra `accuracy_total` dentro de `per_class`.

## Evidencia

- [Resumen JSON](../deliverables/evaluacion_tecnica_90/resumen/resumen_evaluacion.json).
- [Reporte visual existente](../deliverables/evaluacion_tecnica_90/resumen/reporte_prueba_tecnica.png).
- [Script de evaluación](../scripts/evaluate_technical_test.py).
- [Configuración geométrica](../configs/analytic_config.yaml).

En la matriz, las filas son referencias y las columnas predicciones. Ocho errores son referencias centradas clasificadas a izquierda (6) o derecha (2); dos son referencias izquierdas clasificadas al centro. No hubo rechazos: esta muestra no mide por sí sola la capacidad de detectar todos los casos no evaluables.

## Tres niveles de reproducción

1. **Inspeccionar resultados:** comprobar conteos y métricas del JSON sin ejecutar el modelo.
2. **Ejecutar el software:** la demo sintética prueba limpieza, geometría, reglas y exportación, sin fotografías reales ni segmentación.
3. **Repetir la evaluación:** requiere imágenes, referencias, pesos y configuraciones correspondientes. El conjunto local `data/benchmarks/prueba_tecnica_90/` no está versionado. No se promete reproducción pública completa sin estos archivos y sus permisos de acceso.

## Límites y comportamiento conocido

La geometría se refiere a la propia máscara, no al centro de masa, volumen o estabilidad del CAEX. La exactitud por imagen tampoco valida automáticamente la agregación por video.

En `rules.py`, un empate de votos laterales favorece izquierda. En `decide_video_verdict`, las medianas contradictorias pueden favorecer izquierda porque esa condición se comprueba primero. Las pruebas nuevas preservan el comportamiento actual como referencia. Un cambio posterior debe versionarse y acompañarse de una nueva evaluación, sin sustituir silenciosamente las cifras anteriores.

Las pruebas automatizadas verifican software; no aumentan ni certifican el rendimiento en imágenes reales. Una evaluación futura debería ampliar condiciones de captura y separar secuencias para estudiar escenas diferentes.
