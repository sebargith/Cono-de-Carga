# Revisión de publicación

Un archivo versionado no demuestra autorización para distribuirlo. No se han eliminado archivos ni reescrito commits al preparar el portafolio.

| Material localizado | Comprobación pendiente |
| --- | --- |
| Pesos en `perception/models/primary/` y `alternative/` | Confirmar condiciones de distribución de cada artefacto; la aplicación usa el principal. |
| Imágenes en `perception/model_testing/reference/` | Confirmar permisos sobre fotografías operacionales e identificadores visibles. |
| Figuras e informes en `deliverables/` | Confirmar difusión de capturas, información de operación, documentos y marcas. |
| Código y dependencias | Confirmar derechos del autor y de la empresa antes de elegir una licencia. No se añade una licencia sin esa confirmación. |
| Datos locales en `data/` | No constituyen un conjunto público de evaluación. Revisar cualquier futura publicación por separado. |

La portada reutiliza figuras ya versionadas; no incorpora nuevas fotografías de operación. La demo y sus pruebas generan geometrías sintéticas sin archivos de la empresa. Los resultados numéricos provienen del resumen existente.

Es un inventario inicial, no una certificación de permisos ni una auditoría completa de secretos o del historial. Si algún archivo no puede difundirse, se debe acordar su retiro y evaluar también las versiones anteriores en Git, sin borrados automáticos.

## Presentación en GitHub

Descripción sugerida:

> Mining haul truck load-shape analysis with segmentation, geometric rules, video summaries and a Streamlit review app. Includes a CPU-only synthetic demo.

Temas: `computer-vision`, `mining`, `image-segmentation`, `opencv`, `python`, `streamlit`, `geometric-analysis`, `machine-learning`.

La portada `README.md` está en inglés y enlaza la versión equivalente `README.es.md`. `README.en.md` conserva un enlace para lectores anteriores. No añadir una insignia de CI aprobado sin comprobar una ejecución remota. La descripción y los temas son ajustes de GitHub: este archivo los prepara, pero no cambia metadatos remotos.
