# Herramientas del informe

- `generate_report_figures.py`: regenera las figuras C.1 a C.11 usando las
  fuentes consolidadas en `deliverables/informe_final/figuras_fuente/`.
- `update_final_report.py`: actualiza las referencias del modelo de segmentación y
  reemplaza la figura C.4 en el documento existente.
- `render_docx_windows.py`: renderiza el DOCX a PNG para control visual.

Ejecución:

```powershell
cd D:\Memoria\Cono
python tools\reporting\generate_report_figures.py
python tools\reporting\update_final_report.py
```

El documento final se mantiene en:

```text
deliverables/informe_final/Parte_Cono_de_Carga_Informe_Final.docx
```
