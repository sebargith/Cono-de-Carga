# Web App Local - Cono de Carga

Aplicacion local simple para operar el pipeline completo desde videos subidos por el usuario.

La app queda encapsulada en este directorio:

```text
web_app/
  app.py
  src/
  data/
  outputs/
```

`data/` y `outputs/` no se versionan.

## Requisitos

- Python 3.12.
- Controlador NVIDIA compatible para usar `device=0`.
- Dependencias de `requirements/runtime.txt`.

Docker no es necesario para abrir la app ni para ejecutar la inferencia. Se
mantiene únicamente en `perception/` para entrenamiento y validación de
modelos.

## Instalacion

Desde la raiz del proyecto:

```powershell
cd D:\Memoria\Cono
py -3.12 -m venv web_app\.venv
.\web_app\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\runtime.txt
```

La instalación incorpora Streamlit, PyTorch y Ultralytics. Puede tardar varios
minutos la primera vez. Para comprobar que la GPU quedó disponible:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Ejecutar

```powershell
cd D:\Memoria\Cono
.\web_app\.venv\Scripts\Activate.ps1
streamlit run web_app\app.py --server.address 127.0.0.1 --server.port 8502
```

También se puede usar `Abrir web app Cono de Carga.bat`. El lanzador crea
`web_app\.venv` cuando falta, instala las dependencias una sola vez y luego
inicia Streamlit.

La aplicacion de Cono usa de forma dedicada:

```text
http://127.0.0.1:8502
```

Esto evita conflictos con otras aplicaciones Streamlit del proyecto Memoria
que pueden estar ejecutandose en el puerto `8501`.

La app guarda:

```text
web_app\data\uploads\<run_name>\
web_app\outputs\runs\<run_name>\
```

Cada corrida contiene:

```text
work\
  frames\
  yolo_masks\
results\
  analytic\
  manifests\
  statistics\
    indicator_summary.csv
    indicator_distributions.csv
  video_summary.csv
  pipeline_summary.json
```

La seccion `Estadisticas` muestra para los frames validos:

- `peak_x_norm`: posicion relativa del pico principal;
- `area_symmetry`: diferencia relativa entre las areas visibles derecha e izquierda;
- `centroid_x_norm`: posicion relativa del centroide de la mascara.

La interfaz muestra un valor representativo por indicador: la mediana de los
frames validos, acompañada por una lectura directa y una escala horizontal. Los
percentiles P25/P75 y los histogramas tabulados se conservan en los CSV
descargables para analisis detallado. Los valores describen la forma visible
dentro de la propia carga segmentada; no representan centro de masa real ni
balance fisico del camion.

Las tarjetas usan los umbrales guardados en `pipeline_summary.json`:

- `peak_x_norm`: rango `-0.5` a `0.5`; con la configuracion actual, valores
  menores que `-0.06` aportan hacia izquierda y mayores que `0.06` hacia derecha;
- `area_symmetry`: rango `-1` a `1`; valores menores que `-0.07` aportan hacia
  izquierda y mayores que `0.07` hacia derecha;
- `centroid_x_norm`: rango descriptivo `-0.5` a `0.5`, donde el signo indica el
  lado respecto del centro interno de la mascara.

Debajo de las tarjetas se muestra el tiempo de inferencia reportado por
Ultralytics en milisegundos por frame y acumulados. Tambien se informa el tiempo
completo de generacion de mascaras, que incluye prediccion, postproceso y
escritura.
