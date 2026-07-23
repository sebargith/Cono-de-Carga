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

El contenedor YOLO debe existir. Si no esta levantado:

```powershell
cd D:\Memoria\Cono\perception
docker compose up -d
```

Si el contenedor ya estaba creado antes de agregar la web app, recrealo una vez para que tome el montaje nuevo:

```powershell
cd D:\Memoria\Cono\perception
docker compose up -d --force-recreate
```

El `docker-compose.yml` monta `web_app/` dentro del contenedor en:

```text
/workspace/web_app
```

## Instalacion

Desde la raiz del proyecto:

```powershell
cd D:\Memoria\Cono
py -3.12 -m venv web_app\.venv
.\web_app\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r web_app\requirements.txt
```

## Ejecutar

```powershell
cd D:\Memoria\Cono
.\web_app\.venv\Scripts\Activate.ps1
streamlit run web_app\app.py --server.address 127.0.0.1 --server.port 8502
```

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

Para cada indicador se informan P25, mediana y P75, junto con su histograma. Los
valores describen la forma visible dentro de la propia carga segmentada; no
representan centro de masa real ni balance fisico del camion.

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
