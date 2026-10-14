# Ejecución y reproducción

## Demo sin GPU

Desde la raíz del repositorio, con Python 3.12:

```powershell
python -m venv .venv-demo
.\.venv-demo\Scripts\Activate.ps1
python -m pip install -r requirements/demo.txt
python scripts/demo_synthetic.py
python -m unittest discover -s tests -v
```

En Linux/macOS: `source .venv-demo/bin/activate`. Si PowerShell impide activar, usar directamente `.\.venv-demo\Scripts\python.exe` en los comandos, sin cambiar la política de ejecución del equipo.

La demo produce cuatro PNG y `results.csv` en `outputs/demo_synthetic/`: izquierda, centrado, derecha y máscara vacía no evaluable. Para repetir: `python scripts/demo_synthetic.py --output-dir outputs/otra_demo`. Una carpeta no vacía se rechaza para conservar sus resultados.

## Imágenes y máscaras existentes

Con las dependencias de demo y pares imagen/máscara del mismo tamaño y nombre base:

```powershell
python scripts/run_simple_analytic.py --images-dir "<imagenes>" --masks-dir "<mascaras>" --output-dir "outputs/analitico_manual" --config configs/analytic_config.yaml
```

El contrato actual usa carga RGB `[102,255,102]` y fondo negro. Este modo no ejecuta segmentación.

## Flujo completo desde videos

Usar un entorno separado para no mezclar OpenCV headless con el runtime completo:

```powershell
py -3.12 -m venv web_app/.venv
.\web_app\.venv\Scripts\python.exe -m pip install -r requirements/runtime.txt
.\web_app\.venv\Scripts\python.exe scripts/run_pipeline_from_videos.py --videos-dir "data/pipeline_input/videos" --run-name "mi_video" --interval-seconds 0.5 --device 0
```

El runtime fija PyTorch para CUDA 13.0. La ejecución indicada requiere una GPU NVIDIA y un controlador compatibles; no es una receta universal para CPU o macOS. Se espera el modelo `perception/models/primary/psinet_carga_visible.pt` y la clase de carga `material` (`class_id=1`). Verificar disponibilidad y permisos de pesos y videos.

Resultados: `outputs/pipeline_runs/<run_name>/`. La interfaz conserva el lanzador Windows `Abrir web app Cono de Carga.bat`. Docker se utiliza en entrenamiento, no es necesario para inferencia local. [Guía técnica](pipeline_videos_locales.md).

## Comprobaciones

La demo y las pruebas se comprueban localmente en Windows con un entorno limpio de Python 3.12. El workflow `.github/workflows/tests.yml` prepara pruebas de CPU en Windows y Ubuntu. Su resultado remoto debe verificarse después de publicar: añadir el archivo no significa que ya se haya ejecutado. No se ha vuelto a entrenar el modelo ni repetido la evaluación industrial para estos cambios.
