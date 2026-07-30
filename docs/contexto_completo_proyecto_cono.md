# Contexto completo del proyecto Cono de Carga

## 1. Propósito de este documento

Este archivo reúne el estado técnico, las decisiones, los experimentos, los resultados y las limitaciones conocidas del proyecto **Cono de Carga en CAEX**. Está escrito como documento de transferencia de contexto para que otra IA o persona pueda continuar el trabajo sin depender del historial del chat.

Fecha de consolidación: **2026-07-30**.

Raíz local del proyecto:

```text
D:\Memoria\Cono
```

Repositorio remoto:

```text
https://github.com/sebargith/Cono-de-Carga
```

`D:\Memoria\Cono` es también la raíz Git. La antigua copia oculta de
sincronización fue eliminada para evitar dos árboles de trabajo divergentes.
Antes de modificar el proyecto se debe revisar la rama activa y `git status`
directamente en esta raíz.

### 1.1 Instituciones y contexto industrial

El proyecto se desarrolla con la participación o dentro del contexto de tres organizaciones principales, cuyos roles deben distinguirse:

#### Universidad de Concepción - Facultad de Ingeniería

La **Universidad de Concepción (UdeC)** es la institución académica donde se desarrolla esta Memoria de Título, específicamente en su **Facultad de Ingeniería**. La UdeC fue fundada en 1919 y se define institucionalmente como una universidad laica y pluralista, orientada a la formación de personas y a la generación, preservación y transferencia de conocimiento. En este proyecto aporta el marco académico, metodológico y de evaluación propio de una Memoria de Título.

La Facultad de Ingeniería constituye la unidad académica desde la cual se aborda el problema mediante visión computacional, procesamiento de imágenes, aprendizaje automático y desarrollo de software aplicado. El trabajo se alinea con su actividad de investigación, innovación y transferencia de conocimiento aplicado hacia sectores estratégicos.

Fuentes oficiales:

- [Sobre la Universidad de Concepción](https://www.udec.cl/sobre-la-udec/)
- [Facultad de Ingeniería de la Universidad de Concepción](https://www.udec.cl/facultades/facultad-de-ingenieria/)
- [Investigación, innovación y transferencia en Ingeniería UdeC](https://fi.udec.cl/conocimiento-que-se-protege-innovacion-que-se-proyecta-ingenieria-udec/)

#### PSINet

**PSINet** es la empresa colaboradora y contraparte técnica del proyecto. Su actividad se concentra en soluciones e integraciones tecnológicas para minería. Según su información institucional, el Grupo PSINet opera desde 2001 y trabaja en infraestructura tecnológica, redes industriales, servicios IT y OT, monitoreo, control y analítica, además de telecomunicaciones y automatización minera.

En el proyecto Cono de Carga, PSINet vincula la investigación con la aplicación
industrial: está asociada al acceso y contexto de los datos y a la orientación
práctica de la solución. El pipeline utiliza un único modelo entrenado de
segmentación de carga, versionado en `perception/models/primary/`, como etapa
auxiliar para generar las máscaras que consume el analítico geométrico.

Fuente oficial:

- [Quiénes somos - PSINet](https://psinet.cl/nosotros/)

#### Compañía Minera Doña Inés de Collahuasi

La **Compañía Minera Doña Inés de Collahuasi** es la operación minera de la cual provienen los datos utilizados en este proyecto, según el contexto entregado por sus responsables. Collahuasi se dedica a la extracción y producción de concentrado de cobre y molibdeno. Sus operaciones se ubican en la Región de Tarapacá y comprenden los yacimientos a cielo abierto Rosario y Ujina, situados en la comuna de Pica a una altura promedio aproximada de 4.400 metros sobre el nivel del mar.

Para este trabajo, su relevancia es operacional: las imágenes y videos muestran camiones CAEX en condiciones reales de faena y permiten estudiar la forma visible de la carga y la granulometría visible en tolva. Que los datos procedan de esta operación no debe interpretarse automáticamente como una participación académica directa o un convenio formal de Collahuasi en la Memoria, salvo que exista documentación institucional adicional que lo confirme.

Fuentes oficiales:

- [Nuestra compañía - Collahuasi](https://www.collahuasi.cl/quienes-somos/nuestra-compania/)
- [Nuestra operación - Collahuasi](https://www.collahuasi.cl/que-hacemos/nuestra-operacion/)

#### Relación resumida

```text
Universidad de Concepción / Facultad de Ingeniería
  -> institución académica y marco de la Memoria de Título

PSINet
  -> empresa colaboradora y contraparte técnica

Collahuasi
  -> contexto operacional minero y origen de los datos analizados
```

Información institucional consultada el **2026-07-21**.

## 2. Resumen ejecutivo

El proyecto analiza la **forma visible de la carga** transportada por camiones CAEX observados desde atrás. El objetivo es determinar si la forma segmentada se observa centrada, con tendencia visual interna hacia la izquierda o con tendencia visual interna hacia la derecha.

La solución quedó dividida en cuatro capacidades:

1. Extracción de frames desde videos.
2. Segmentación automática de la carga mediante YOLO-seg.
3. Analítico geométrico sobre la máscara de carga.
4. Presentación de resultados por frame y veredicto agregado por video.

Flujo principal:

```text
video local
-> extracción de frames
-> YOLO-seg
-> máscara RGB compatible con CVAT
-> limpieza de máscara
-> features geométricas
-> clase por frame
-> overlays y CSV
-> resumen y veredicto por video
```

También existe una web app local en Streamlit que permite subir videos, ejecutar el pipeline y revisar consola, overlays, estadísticas, veredicto y archivos descargables.

El resultado cuantitativo principal obtenido hasta ahora corresponde al
**rendimiento del analítico geométrico completo**, alimentado por las máscaras
del modelo entrenado de segmentación y usando los umbrales analíticos actuales.
En una prueba balanceada de 90 imágenes obtuvo:

```text
80/90 correctas
88,89 % de exactitud total
90/90 imágenes evaluables
```

Este 88,89 % mide la concordancia del analítico con las clases visuales
esperadas (`centrado`, `izquierda` y `derecha`). No es una métrica aislada del
modelo de segmentación ni una medición física del balance, centro de masa o
distribución real del payload.

El modelo de segmentación ya está conectado al pipeline general y a la
aplicación web. La configuración selecciona `class_id=1` por defecto; el
parámetro también puede sobrescribirse desde los orquestadores.

La etapa de segmentación no es exclusiva de Cono. El mismo archivo binario se
utiliza en Granulometría para aislar la superficie visible del material. Ambos
subproyectos comparten, por tanto, un único modelo y la semántica de la clase
`material`, pero aplican adaptadores y postprocesamientos distintos antes de
ejecutar sus analíticos específicos.

## 3. Alcance conceptual y terminología

### 3.1 Qué se mide

El analítico describe geometría relativa a la propia máscara segmentada:

- forma visible de la carga;
- simetría interna izquierda/derecha;
- pico relativo respecto del centro del bbox de la máscara;
- irregularidad del perfil superior;
- asimetría visual interna;
- distribución de píxeles segmentados a ambos lados del centro del bbox.

### 3.2 Qué no se mide

No se debe presentar el resultado como:

- centro de masa real;
- balance físico del camión;
- desplazamiento respecto de la tolva;
- distribución real del payload;
- medición volumétrica o de tonelaje.

No existe todavía máscara de tolva ni ROI geométrica estable. Por eso todas las referencias son internas a la máscara de carga. Si la segmentación queda incompleta, también cambian el bbox, su centro y las features normalizadas.

### 3.3 Clases actuales

Las clases por frame son:

```text
centrado
izquierda
derecha
no_evaluable
```

La clase `doble_pico` fue eliminada como salida final. `num_peaks` continúa existiendo como feature diagnóstica, pero no produce una clase independiente.

La irregularidad tampoco es una clase final en la versión actual. `profile_roughness` se conserva como feature descriptiva.

## 4. Evolución resumida

### Mayo de 2026

- Se definió el formato CVAT y la lectura de máscaras RGB.
- Se implementó el analítico geométrico inicial.
- Se agregó el script de ejecución y los overlays.

### 2 de junio de 2026

- Se agregó descarte de máscaras que tocan bordes del frame.
- Se creó el módulo Docker/GPU para entrenamiento YOLO-seg.
- Se implementó preparación de datasets CVAT -> YOLO-seg.
- Se integró MLflow.
- Se agregaron validación, inferencia, combinación de ZIPs CVAT y un modelo de referencia.

### 3 de junio de 2026

- Se incorporó Frame Hunter para Nextcloud.
- Se creó el adaptador YOLO -> máscaras compatibles con el analítico.
- Se creó el pipeline desde videos locales.
- Se corrigió la detección de múltiples picos falsos en mesetas superiores.

### 4 y 5 de junio de 2026

- Se agregó veredicto agregado por video.
- Se ajustó sensibilidad lateral.
- Se creó la web app local Streamlit.
- Se agregó consola en vivo y se corrigió su duplicación.

### 25 de junio de 2026

- Se agregó procesamiento masivo de imágenes.
- Se agregó un lanzador `.bat` para abrir la web app con doble clic.

### 16 y 17 de julio de 2026

- Se creó la evaluación cuantitativa de prueba técnica.
- Se continuó el ajuste de umbrales, pesos y sensibilidad del analítico mediante
  varias pruebas de desarrollo, no restringidas al conjunto de 90 imágenes.
- Se agregaron reportes, matriz de confusión y comparaciones visuales por clase.

### 23 de julio de 2026

- Se repitió la evaluación de rendimiento del analítico sobre las 90 imágenes.
- Se reprodujo el resultado de 88,89 % de exactitud del analítico.
- Se incorporaron indicadores geométricos resumidos y distribuciones exportables.

### 30 de julio de 2026

- Se reorganizaron código, modelos, resultados y entregables del repositorio.
- Se consolidó un único modelo entrenado de segmentación como etapa auxiliar
  operacional; los checkpoints de desarrollo quedaron solo como antecedentes.
- Se migró la inferencia de YOLO desde Docker hacia el entorno Python local.
- Docker quedó reservado para preparación, entrenamiento, validación y MLflow.
- Se incorporaron pruebas automatizadas de inferencia local y estadísticas.

## 5. Estructura relevante

```text
D:\Memoria\Cono\
  configs\
    analytic_config.yaml
  src\cone_load\
    io.py
    masks.py
    profile.py
    features.py
    rules.py
    visualization.py
  scripts\
    run_simple_analytic.py
    run_pipeline_from_videos.py
    run_manager_sample_from_images.py
    evaluate_technical_test.py
    create_technical_comparisons.py
  perception\
    Dockerfile
    docker-compose.yml
    configs\perception_config.yaml
    scripts\
      prepare_yolo_dataset.py
      train_yolo_seg.py
      validate_yolo_seg.py
      predict_yolo_seg.py
      predict_to_analytic_masks.py
      check_env.py
    src\cone_perception\
    tools\cvat_zip_merger\
    data\raw\
    data\yolo\
    runs\
    models\
    model_testing\
    mlruns\
  tools\
    frame_hunter_nextcloud\
    reporting\
  web_app\
    app.py
    src\pipeline_runner.py
    src\run_storage.py
    data\
    outputs\
  data\
  deliverables\
  docs\
  outputs\
  Abrir web app Cono de Carga.bat
```

Los datos, resultados, entornos virtuales, cachés y la mayoría de los modelos experimentales no deben tratarse como código fuente.

## 6. Datos y formato CVAT

### 6.1 Exportación esperada

La exportación CVAT usada es **Segmentation Mask**:

```text
labelmap.txt
JPEGImages\
SegmentationClass\
```

Colores definidos:

```text
Carga:102,255,102::
background:0,0,0::
```

El analítico busca exactamente RGB `[102, 255, 102]`, con tolerancia actual `0`. Las máscaras deben guardarse en PNG, BMP o TIFF; no se recomienda JPEG porque altera los colores por compresión.

### 6.2 Datos de entrenamiento YOLO inicial

El dataset YOLO preparado contiene 240 muestras:

```text
train: 180
val: 36
test: 24
```

Ruta:

```text
D:\Memoria\Cono\perception\data\yolo\carga_visible_seg
```

El preparador puede:

- respetar carpetas `data/raw/train`, `val` y `test`;
- combinar exportaciones CVAT y hacer split aleatorio;
- convertir contornos de la máscara RGB a polígonos YOLO-seg;
- omitir o incluir etiquetas vacías según configuración;
- escribir `dataset.yaml` y `manifest.csv`.

### 6.3 Combinador de ZIPs CVAT

Herramienta:

```text
perception\tools\cvat_zip_merger\merge_cvat_zips.py
```

Sirve para reunir varios ZIPs por partición en una exportación consolidada antes de preparar YOLO.

## 7. Analítico geométrico

### 7.1 Entrada y emparejamiento

El script principal es:

```text
scripts\run_simple_analytic.py
```

Entradas:

- directorio de imágenes;
- directorio de máscaras;
- configuración YAML;
- directorio de salida.

Imagen y máscara se emparejan por nombre base o `stem`. Los stems deben ser únicos.

### 7.2 Conversión y limpieza de máscara

El flujo es:

```text
máscara RGB
-> detección del color de Carga
-> binaria 0/1
-> apertura morfológica
-> cierre morfológico
-> relleno de huecos
-> componente conectado principal
```

Configuración actual:

```yaml
mask:
  target_rgb: [102, 255, 102]
  background_rgb: [0, 0, 0]
  color_tolerance: 0
  warn_unexpected_color_fraction: 0.0001
  morph_open_kernel: 3
  morph_close_kernel: 5
  fill_holes: true
  keep_largest_component: true
```

Los píxeles con colores distintos de carga y fondo quedan registrados mediante:

```text
unexpected_pixel_count
unexpected_color_count
warnings
```

Si se eliminan componentes secundarios, se registra una advertencia como:

```text
componentes_secundarios_removidos:<N>_px
```

### 7.3 Validaciones

Configuración actual:

```yaml
geometry:
  min_area_px: 100
  min_bbox_width: 10
  min_bbox_height: 10
  min_profile_points: 20
  border_touch_margin_px: 0
```

Motivos principales de `no_evaluable`:

```text
imagen_sin_mascara_correspondiente
mascara_vacia
mascara_muy_pequena
bbox_demasiado_pequeno
perfil_superior_insuficiente
mascara_toca_borde_imagen
```

Una máscara que toca cualquier borde se conserva en el CSV, pero queda:

```text
predicted_class = no_evaluable
is_valid = False
error_reason = mascara_toca_borde_imagen
```

Se guardan flags específicos para borde izquierdo, derecho, superior e inferior.

### 7.4 Bounding box y centro interno

El bbox es el rectángulo mínimo que contiene todos los píxeles de la máscara limpia.

Importante: el bbox no conoce la tolva. Si YOLO omite una parte de la carga, el bbox se ajusta correctamente a una máscara incorrecta o incompleta. Esto cambia el centro interno y puede distorsionar todas las normalizaciones.

### 7.5 Centroide

Se calcula con momentos binarios de OpenCV.

```text
centroid_x_norm = (centroid_x - bbox_center_x) / bbox_width
```

El centroide no se usa actualmente para decidir la clase; queda como feature descriptiva.

### 7.6 Perfil superior

Para cada columna con carga se obtiene el primer píxel desde arriba:

```text
y_top(x) = primer píxel de carga en la columna x
```

Suavizado actual:

```yaml
profile:
  smoothing:
    method: savgol
    window_length: 31
    polyorder: 2
    moving_average_window: 15
```

Se usa Savitzky-Golay cuando SciPy está disponible. Si no está disponible, se usa media móvil.

Detección de picos:

```yaml
peaks:
  prominence_px: 8.0
  min_distance_px: 20
  apex_plateau_tolerance_px: 6.0
```

El pico principal corresponde al mínimo de `y` del perfil suavizado. Las mesetas cercanas al ápice se colapsan hacia el centro para evitar múltiples picos falsos.

```text
peak_x_norm = (peak_x - bbox_center_x) / bbox_width
```

Valores negativos apuntan a izquierda interna; positivos, a derecha interna.

### 7.7 Simetría de área

La máscara se divide usando el centro del bbox:

```text
area_symmetry = (area_right - area_left) / (area_right + area_left)
```

- negativo: más área segmentada a la izquierda;
- positivo: más área segmentada a la derecha;
- cercano a cero: simetría interna aproximada.

### 7.8 Irregularidad del perfil

```text
profile_roughness = std(y_raw - y_smooth) / bbox_height
```

Es una irregularidad normalizada por la altura del bbox. Actualmente no decide la clase.

### 7.9 Reglas actuales

Configuración vigente:

```yaml
rules:
  threshold_peak: 0.06
  threshold_area: 0.07
  peak_vote_weight: 0.5
  area_vote_weight: 0.5
```

Reglas:

- `peak_x_norm < -0.06`: suma voto izquierda.
- `peak_x_norm > 0.06`: suma voto derecha.
- `area_symmetry < -0.07`: suma voto izquierda.
- `area_symmetry > 0.07`: suma voto derecha.
- sin votos: `centrado`.
- mayor puntaje izquierdo: `izquierda`.
- mayor puntaje derecho: `derecha`.

Detalle importante: con pesos `0.5/0.5`, si pico y área votan lados opuestos se produce empate. El código usa:

```python
if left_score >= right_score:
    return "izquierda"
```

Por lo tanto, todo empate contradictorio se resuelve como izquierda. Es un sesgo conocido y debería revisarse; alternativas posibles son `centrado`, `no_evaluable`, usar una magnitud continua o definir una regla explícita de desempate.

### 7.10 Features del CSV

Columnas principales:

```text
image_name
mask_area_px
bbox_xmin
bbox_ymin
bbox_xmax
bbox_ymax
bbox_width
bbox_height
bbox_aspect_ratio
touches_left_border
touches_right_border
touches_top_border
touches_bottom_border
is_truncated_by_frame
centroid_x
centroid_y
centroid_x_norm
peak_x
peak_y
peak_x_norm
area_left
area_right
area_symmetry
profile_roughness
num_peaks
predicted_class
is_valid
```

Trazabilidad adicional:

```text
error_reason
warnings
mask_path
overlay_path
unexpected_pixel_count
unexpected_color_count
```

### 7.11 Overlays

Cada overlay muestra:

- imagen original;
- máscara verde superpuesta;
- bbox amarillo;
- eje central del bbox;
- centroide;
- perfil superior suavizado;
- pico principal;
- picos detectados;
- clase;
- `peak_x_norm`;
- `area_symmetry`;
- `profile_roughness`;
- `num_peaks`;
- error cuando el frame no es evaluable.

## 8. Segmentación auxiliar con YOLO

### 8.1 Entornos de ejecución

El módulo se encuentra en:

```text
D:\Memoria\Cono\perception
```

Docker con GPU NVIDIA se utiliza para preparación de datos, entrenamiento,
validación y registro en MLflow. La imagen se basa en:

```text
nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
Python 3.10
PyTorch/Torchvision/Torchaudio CUDA 12.4
Ultralytics
MLflow
boto3
OpenCV headless
NumPy
SciPy
PyYAML
```

El cambio de ubicación del almacenamiento interno de imágenes Docker desde C: a D: no modifica comandos del proyecto; Docker Desktop administra esa ubicación.

La inferencia operacional se ejecuta fuera de Docker con el entorno local:

```text
web_app\.venv
```

Sus dependencias se declaran en `requirements/runtime.txt`. El mismo Python
ejecuta Streamlit, `predict_to_analytic_masks.py`, el analítico y la agregación
por video.

### 8.2 Docker Compose para entrenamiento

Servicio principal:

```text
cone-yolo
```

Contenedor:

```text
cono-yolo-perception
```

Las carpetas de configuración, scripts, datos, runs, modelos, pruebas, MLflow y cachés se montan como volúmenes. Cambiar scripts, datos o YAML no exige reconstruir la imagen. Cambiar Dockerfile o dependencias sí exige `docker compose build`.

Comandos básicos:

```powershell
cd D:\Memoria\Cono\perception
docker compose build
docker compose up -d
docker compose exec cone-yolo python scripts/check_env.py
docker compose down
```

### 8.3 MLflow

El entrenamiento registra:

- parámetros;
- configuración;
- dataset y manifest;
- métricas finales y mejores;
- `best.pt` y `last.pt` como artefactos.

La configuración puede usar MLflow local o remoto. Existe un archivo local:

```text
perception\.env.mlflow
```

Contiene URI y credenciales. No copiar sus secretos a documentación ni Git.

`boto3` es obligatorio cuando MLflow guarda artefactos en S3/MinIO.

### 8.4 Antecedente de entrenamiento YOLO

Run:

```text
carga_visible_yolov8n-seg_20260527_195509
```

Arquitectura y parámetros:

```text
yolov8n-seg.yaml
pretrained = false
epochs = 100
batch = 8
imgsz = 640
device = 0
optimizer = auto
patience = 30
```

El entrenamiento fue desde arquitectura, no fine-tuning de un `.pt`.

Métricas registradas en validación:

```text
mejor masks mAP50-95: 0.94142 en epoch 87
epoch final masks mAP50: 0.995
epoch final masks mAP50-95: 0.91647
```

Estas métricas provienen de un conjunto de validación pequeño de 36 imágenes y no equivalen al desempeño end-to-end de clasificación geométrica en cámaras o fechas distintas.

Checkpoint de desarrollo conservado:

```text
perception\models\alternative\best.pt
```

SHA256 conocido:

```text
11BF01302CABD3EAE1CFE65FDAD12B08FB1AA750493D976148CEABA9D35E336C
```

Las copias redundantes de este archivo y de `last.pt` fueron eliminadas. Se
conserva un solo `best.pt` como antecedente reproducible del entrenamiento
realizado en el repositorio. No se considera un segundo modelo del proyecto ni
forma parte del flujo analítico vigente.

### 8.5 Modelo entrenado de segmentación utilizado

Archivo local:

```text
perception\models\primary\psinet_carga_visible.pt
```

Tamaño:

```text
54.829.963 bytes
```

SHA256:

```text
BD7C9880D9A39B291DC03669DAD774DB9FD933D6026DF694EEE1E3A41CBF1DAB
```

Clases internas:

```text
0: caex
1: material
```

Para segmentar carga se selecciona:

```text
--class-id 1
```

Este valor se encuentra en `prediction.class_id` dentro de
`perception/configs/perception_config.yaml`.

Este es el único modelo de segmentación considerado por el flujo vigente. El
archivo está versionado en la rama `main`; su nombre y ubicación se conservan
por compatibilidad y reproducibilidad técnica.

#### Correspondencia con Granulometría

El modelo de segmentación de Cono es binariamente idéntico al utilizado por el
proyecto de Granulometría:

```text
Cono:
D:\Memoria\Cono\perception\models\primary\psinet_carga_visible.pt

Granulometría:
D:\Memoria\Granulometria\modelos percepcion\carga_tolva_yolo\
haul_truck_segment_m_736_low_data_aug_07012025.pt
```

Ambos archivos tienen:

```text
tamaño: 54.829.963 bytes
SHA256: BD7C9880D9A39B291DC03669DAD774DB9FD933D6026DF694EEE1E3A41CBF1DAB
clases: 0=caex, 1=material
```

Que el modelo sea común no significa que exista todavía un único paquete de
segmentación compartido. Cada repositorio conserva su copia para mantener el
pipeline encapsulado y dispone de su propio adaptador. Antes de centralizar el
peso o el código se deben estabilizar las rutas, el contrato de salida y las
configuraciones de ambos consumidores.

### 8.6 Configuración específica de cada consumidor

El modelo común de segmentación se usa actualmente con parámetros distintos:

| Parámetro | Cono de Carga | Granulometría |
|---|---:|---:|
| Modelo de segmentación | mismo SHA256 | mismo SHA256 |
| Clase de material | `1` | `1` |
| `imgsz` predeterminado | `640` | `736` |
| Confianza predeterminada | `0.35` | `0.25` |
| Selección de instancias | `largest` | `union` |
| IoU explícito | valor por defecto de Ultralytics | `0.7` |

En Cono, `largest` reduce el riesgo de incorporar acopios exteriores como parte
de la carga. El adaptador genera una máscara RGB compatible con CVAT y el
analítico. Granulometría puede unir instancias antes de recortar, normalizar y
dividir la carga en patches. Estas decisiones pertenecen a los consumidores de
la segmentación y deben mantenerse separadas de la identidad del checkpoint.

El rendimiento de 88,89 % del analítico de Cono se obtuvo con máscaras
generadas usando `imgsz=640`, `conf=0.35`, `class_id=1` y `largest`. Estos datos
se conservan para reproducir la evaluación, no para establecer una comparación
entre modelos. No se deben unificar parámetros con Granulometría sin validar el
efecto sobre las máscaras y los resultados de ambos analíticos.

### 8.7 Adaptador YOLO -> analítico

Script:

```text
perception\scripts\predict_to_analytic_masks.py
```

Responsabilidades:

- cargar un modelo Ultralytics de segmentación;
- filtrar por `class_id`;
- rasterizar polígonos al tamaño original;
- elegir instancia principal o unir instancias;
- generar máscara RGB `[102,255,102]` sobre fondo negro;
- guardar `manifest.csv` y `summary.json`.

Modos:

```text
largest   -> conserva la instancia de mayor área
merge_all -> une todas las instancias de la clase seleccionada
```

`largest` es el modo por defecto y es más seguro sin ROI de tolva. En una inspección se comprobó que `merge_all` podía incorporar un acopio exterior detectado erróneamente como material.

El `conf` de YOLO decide qué instancias se aceptan; no expande los píxeles internos de una máscara ya detectada.

## 9. Pipeline desde videos locales

Orquestador:

```text
scripts\run_pipeline_from_videos.py
```

Etapas:

1. Busca videos locales `.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`.
2. Extrae frames con OpenCV.
3. Guarda manifiestos de videos y frames.
4. Ejecuta YOLO con el entorno Python local.
5. Convierte predicciones a máscaras RGB.
6. Ejecuta el analítico con Python local.
7. Genera features y overlays.
8. Agrupa resultados por video.
9. Genera un veredicto final por video.

Parámetros relevantes:

```text
--interval-seconds 2
--interval-seconds 0
--max-frames-per-video
--limit-videos
--recursive
--conf 0.35
--imgsz
--device 0
--mask-mode largest
--video-min-valid-ratio 0.5
--video-majority-ratio 0.6
```

Múltiples videos en la carpeta se procesan dentro de la misma corrida, manteniendo relación frame-video mediante manifests.

Salidas típicas:

```text
perception\model_testing\pipeline_runs\<run_name>\
  frames\
  yolo_masks\

outputs\pipeline_runs\<run_name>\
  analytic\tables\features.csv
  analytic\overlays\
  manifests\videos.csv
  manifests\frames.csv
  manifests\yolo_manifest.csv
  manifests\yolo_summary.json
  statistics\indicator_summary.csv
  statistics\indicator_distributions.csv
  video_summary.csv
  pipeline_summary.json
```

Desde julio de 2026, el pipeline agrega un resumen estadistico de los tres
indicadores geometricos mas descriptivos: `peak_x_norm`, `area_symmetry` y
`centroid_x_norm`. `indicator_summary.csv` registra cantidad, P25, mediana y
P75 por corrida y por video; `indicator_distributions.csv` registra
histogramas de 20 intervalos. Se calculan exclusivamente con frames validos y
describen la forma visible de la carga relativa a su propia segmentacion.

La aplicacion presenta ademas los rangos numericos de interpretacion. Con la
configuracion vigente, `peak_x_norm` usa `-0.06` y `0.06` como referencias
laterales, mientras que `area_symmetry` usa `-0.07` y `0.07`.
`centroid_x_norm` se informa en su rango teorico `[-0.5, 0.5]` como indicador
descriptivo y no como regla independiente del veredicto.

Para mantener una lectura simple, la interfaz presenta solo la mediana de cada
indicador, su interpretacion y una escala horizontal. P25, P75 e histogramas no
se muestran en la vista principal, pero se conservan en los CSV para analisis
estadistico y trazabilidad.

El script YOLO registra los tiempos entregados por Ultralytics: inferencia
media por frame, inferencia acumulada, preproceso y postproceso. Tambien mide el
tiempo de pared de la generacion de mascaras. Estos valores se copian a
`pipeline_summary.json` y se muestran fuera de las tarjetas generales de la
web app.

### 9.1 Veredicto por video

Clases agregadas:

```text
carga_centrada
tendencia_izquierda
tendencia_derecha
mixto_o_inestable
no_evaluable
```

Lógica:

- si la proporción de frames válidos es menor al mínimo, `no_evaluable`;
- si una clase supera el umbral de mayoría, se usa esa clase;
- sin mayoría, se evalúan medianas de `peak_x_norm` y `area_symmetry`;
- si no hay señal clara, `mixto_o_inestable`.

Se guardan conteos, ratios, medianas, medias, confianza y motivo del veredicto.

### 9.2 Modelo de segmentación y clase

El pipeline utiliza el único modelo de segmentación vigente,
`perception/models/primary/psinet_carga_visible.pt`, y
`prediction.class_id: 1` de forma predeterminada. `run_pipeline_from_videos.py`
y `run_manager_sample_from_images.py` exponen `--class-id` y lo propagan al
adaptador. Los checkpoints de desarrollo conservados fuera de esta ruta son
antecedentes históricos y no forman parte del flujo ni de la evaluación actual.

## 10. Web app local

Tecnología:

```text
Streamlit
Pandas
OpenCV
Python local
PyTorch y Ultralytics locales
```

Archivos principales:

```text
web_app\app.py
web_app\src\pipeline_runner.py
web_app\src\run_storage.py
web_app\requirements.txt
```

Funciones:

- carga de uno o varios videos;
- nombre de corrida;
- intervalo de extracción;
- límite de frames;
- confianza YOLO;
- dispositivo;
- consola en vivo y scrolleable;
- selección de video dentro de una corrida;
- overlays;
- veredicto;
- estadísticas y tabla por frame;
- descargas de CSV y JSON.

Orden visual vigente:

```text
título del video
-> consola
-> overlays
-> veredicto
-> estadísticas
-> descargas
```

Lanzador de doble clic:

```text
Abrir web app Cono de Carga.bat
```

El lanzador:

- crea `web_app\.venv` si falta;
- instala Streamlit, PyTorch, Ultralytics y las dependencias del analítico si
  faltan;
- comprueba si CUDA está disponible en el entorno local;
- abre Streamlit en `http://127.0.0.1:8502`;
- utiliza un puerto separado para no interferir con la aplicacion de
  Granulometria, que puede estar ejecutandose en `8501`.

La web app usa el modelo entrenado de segmentación definido en
`run_pipeline_from_videos.py`, con la clase `material` configurada.

## 11. Frame Hunter y videos

Directorio:

```text
tools\frame_hunter_nextcloud
```

Puede:

- conectarse a Nextcloud;
- listar y validar videos específicos;
- descargar videos;
- extraer frames con FFmpeg u OpenCV;
- configurar intervalo y calidad.

Salidas:

```text
tools\frame_hunter_nextcloud\data\videos
tools\frame_hunter_nextcloud\data\frames\<video>
```

El pipeline operativo actual no descarga desde Nextcloud. Toma videos desde una carpeta local. Frame Hunter se mantiene como herramienta separada para abastecer esa carpeta si se necesita.

No copiar credenciales reales a documentación ni Git.

## 12. Evaluación técnica por imágenes

### 12.1 Dataset

Ruta de entrada:

```text
D:\Memoria\Cono\data\benchmarks\prueba_tecnica_90
```

Estructura:

```text
Centrado\   30 imágenes
Izquierda\  30 imágenes
Derecha\    30 imágenes
```

Total: 90 imágenes seleccionadas manualmente por apariencia visual.

Los umbrales y la sensibilidad del analítico se ajustaron mediante varias
pruebas de desarrollo con distintas muestras y corridas; el proceso no se
limitó a estas 90 imágenes. Este conjunto balanceado se utiliza aquí como
evaluación de rendimiento del analítico geométrico con su configuración
vigente.

### 12.2 Herramientas

Procesamiento masivo:

```text
scripts\run_manager_sample_from_images.py
```

Evaluación contra carpetas esperadas:

```text
scripts\evaluate_technical_test.py
```

Generación de composiciones original/overlay:

```text
scripts\create_technical_comparisons.py
```

El evaluador genera:

- `evaluacion_por_imagen.csv`;
- `indicadores_geometricos.csv`, con P25, mediana y P75 por clase esperada;
- `distribuciones_geometricas.csv`, con los valores por imagen usados para estudiar cada distribución;
- `resumen_evaluacion.json`;
- matriz de confusión;
- reporte PNG con exactitud, matriz de confusión e indicadores geométricos;
- originales y overlays por clase esperada.

### 12.3 Antecedentes del ajuste de sensibilidad

El ajuste de parámetros incluyó múltiples pruebas de desarrollo, tanto sobre
la sensibilidad lateral como sobre el equilibrio entre las señales de pico y
área. Entre las configuraciones exploradas estuvieron:

| `threshold_peak` | `threshold_area` | Pesos pico/área |
|---:|---:|---:|
| 0.09 | 0.10 | 0.6 / 0.4 |
| 0.08 | 0.09 | 0.6 / 0.4 |
| 0.06 | 0.07 | 0.5 / 0.5 |

La última combinación quedó vigente porque aumentó la sensibilidad a
tendencias laterales manteniendo un comportamiento aceptable en centrado.
Estas configuraciones documentan parte de un proceso de pruebas más amplio; no
deben interpretarse como un ajuste realizado exclusivamente sobre el conjunto
de 90 imágenes ni como una comparación entre modelos de segmentación.

### 12.4 Rendimiento del analítico geométrico

La evaluación del flujo vigente utilizó:

```text
modelo entrenado de segmentación como etapa auxiliar
class_id = 1 (material)
conf = 0.35
mask_mode = largest
threshold_peak = 0.06
threshold_area = 0.07
pesos = 0.5 / 0.5
```

Resultado de rendimiento del analítico:

```text
90/90 evaluables
80/90 correctas
88,89 % de exactitud
```

Matriz de confusión:

| Esperado | Pred. centrado | Pred. izquierda | Pred. derecha | No evaluable |
|---|---:|---:|---:|---:|
| Centrado | 22 | 6 | 2 | 0 |
| Izquierda | 2 | 28 | 0 | 0 |
| Derecha | 0 | 0 | 30 | 0 |

Exactitud por clase:

```text
Centrado: 22/30 = 73,33 %
Izquierda: 28/30 = 93,33 %
Derecha: 30/30 = 100,00 %
```

Por tanto, el 88,89 % es la exactitud del **analítico geométrico** sobre las 90
imágenes con etiquetas visuales de referencia. El modelo de segmentación
participa únicamente como generador de máscaras dentro del flujo evaluado; este
porcentaje no se presenta como una métrica propia del segmentador.

### 12.5 Repetición con indicadores geométricos

El 23 de julio de 2026 se repitió la evaluación completa con la misma
configuración del analítico para incorporar los nuevos indicadores al reporte.
La corrida reprodujo el resultado de rendimiento:

```text
90/90 imágenes con predicción YOLO
90/90 imágenes evaluables
80/90 clasificaciones correctas
88,89 % de exactitud
```

Los resultados por clase también se reprodujeron:

| Clase esperada | Correctas | Exactitud |
|---|---:|---:|
| Centrado | 22/30 | 73,33 % |
| Izquierda | 28/30 | 93,33 % |
| Derecha | 30/30 | 100,00 % |

El reporte ampliado resume tres indicadores respecto de la clase visual esperada mediante un valor representativo por clase:

| Clase esperada | Pico mediano | Simetría mediana | Centroide mediano |
|---|---:|---:|---:|
| Centrado | -0,004 | 0,024 | 0,007 |
| Izquierda | -0,138 | -0,179 | -0,051 |
| Derecha | 0,109 | 0,172 | 0,050 |

Estos valores muestran una separación consistente entre las tres clases. El centroide presenta menor amplitud que el pico y la simetría de área, por lo que se conserva como indicador descriptivo y no como variable principal de las reglas de decisión.

P25 y P75 permanecen disponibles en `indicadores_geometricos.csv`, pero se
retiraron de la imagen resumen para reducir densidad visual.

Los tiempos técnicos de la etapa auxiliar de segmentación para las 90 imágenes
fueron:

```text
inferencia YOLO acumulada: 1,673 s
inferencia YOLO media: 18,59 ms/imagen
generación completa de máscaras: 20,23 s
```

La generación completa incluye carga de imágenes, preprocesamiento, inferencia, postprocesamiento y escritura de máscaras; por ello es mayor que el tiempo puro de inferencia reportado por YOLO.

### 12.6 Ruta actual de resultados

La repetición con indicadores quedó en:

```text
D:\Memoria\Cono\deliverables\evaluacion_tecnica_90
```

Contiene:

```text
overlays\
comparaciones\Centrado\
comparaciones\Izquierda\
comparaciones\Derecha\
resumen\evaluacion_por_imagen.csv
resumen\indicadores_geometricos.csv
resumen\distribuciones_geometricas.csv
resumen\resumen_evaluacion.json
resumen\reporte_prueba_tecnica.png
```

Los overlays se conservan una sola vez y las comparaciones se separan por clase
esperada. El reporte PNG combina:

- resumen global y exactitud por clase;
- matriz de confusión;
- tabla de indicadores geométricos por clase;
- umbrales analíticos usados.

Algunos CSV pueden contener rutas históricas anteriores al cambio de nombre de carpeta. Si se mueven resultados, regenerar o normalizar paths antes de reutilizar scripts.

## 13. Hallazgos sobre errores de segmentación

### 13.1 El bbox no recupera carga omitida

El bbox se calcula sobre la máscara limpia. Si la máscara omite una franja visible, el bbox queda correctamente ajustado a la máscara, pero no a toda la carga real visible.

Esto puede alterar:

- centro del bbox;
- `peak_x_norm`;
- `centroid_x_norm`;
- `area_left` y `area_right`;
- `area_symmetry`;
- clase final.

### 13.2 `merge_all` no es una solución general

En una imagen inspeccionada, el modelo de segmentación generó dos instancias.
La segunda no era un trozo de carga omitido, sino un acopio exterior.
`merge_all` habría contaminado la máscara y hecho que tocara bordes.

Sin ROI o máscara de tolva, `largest` sigue siendo el modo más prudente.

### 13.3 Casos de subsegmentación

Se observaron omisiones en:

- zonas oscuras u ocluidas;
- material muy claro o sobreexpuesto;
- fragmentos aislados en laterales;
- variaciones de cámara, fecha e iluminación.

Subir `imgsz` a 960 o 1280 puede ayudar con detalle, pero la corrección principal es reentrenar o ajustar con ejemplos representativos y anotaciones consistentes.

La dilatación morfológica grande no es una solución válida: puede inventar carga en sectores vacíos y alterar la geometría.

## 14. Manejo de errores y aspectos operativos

El proyecto contempla:

- rutas inexistentes;
- imágenes sin máscara;
- máscaras vacías;
- colores inesperados;
- bbox pequeño;
- perfil insuficiente;
- máscara truncada por borde;
- stems duplicados;
- salidas existentes sin `--overwrite`;
- modelos o rutas locales inexistentes;
- ausencia de PyTorch o Ultralytics en el entorno local;
- ausencia de predicciones YOLO;
- múltiples instancias.

En Windows se detectó un problema al escribir imágenes en rutas con caracteres Unicode. Las funciones recientes de reportes usan `cv2.imencode(...).tofile(...)`, que soporta mejor estas rutas que `cv2.imwrite`.

En pruebas históricas se utilizó también:

```text
C:\Users\azusu\anaconda3\python.exe
```

El entorno operacional vigente de la web app y del pipeline completo es:

```text
web_app\.venv
```

El pipeline operacional completo se ejecuta en Python local. Docker es
obligatorio únicamente para reproducir la preparación, el entrenamiento y la
validación encapsulada del modelo.

## 15. Comandos de referencia

### 15.1 Analítico con máscaras manuales

```powershell
python D:\Memoria\Cono\scripts\run_simple_analytic.py `
  --images-dir "<JPEGImages>" `
  --masks-dir "<SegmentationClass>" `
  --output-dir "D:\Memoria\Cono\outputs\manual" `
  --config "D:\Memoria\Cono\configs\analytic_config.yaml"
```

### 15.2 Preparar dataset YOLO

```powershell
cd D:\Memoria\Cono\perception
docker compose up -d
docker compose exec cone-yolo python scripts/prepare_yolo_dataset.py --overwrite
```

### 15.3 Reproducir el antecedente de entrenamiento desde arquitectura

```powershell
docker compose exec cone-yolo python scripts/train_yolo_seg.py `
  --config configs/perception_config.yaml `
  --model yolov8n-seg.yaml `
  --no-pretrained `
  --epochs 100 `
  --batch 8 `
  --imgsz 640 `
  --device 0
```

### 15.4 Ejecutar directamente la segmentación

```powershell
cd D:\Memoria\Cono
.\web_app\.venv\Scripts\Activate.ps1

python perception\scripts\predict_to_analytic_masks.py `
  --model "perception\models\primary\psinet_carga_visible.pt" `
  --images-dir "<imagenes>" `
  --masks-dir "<salida_mascaras>" `
  --class-id 1 `
  --conf 0.35 `
  --device 0 `
  --mask-mode largest `
  --overwrite
```

### 15.5 Pipeline desde videos locales

```powershell
cd D:\Memoria\Cono
.\web_app\.venv\Scripts\Activate.ps1
python scripts\run_pipeline_from_videos.py `
  --videos-dir "data\pipeline_input\videos" `
  --run-name "prueba_videos_01" `
  --interval-seconds 2 `
  --conf 0.35 `
  --device 0 `
  --overwrite
```

Este comando usa por defecto el único modelo entrenado de segmentación del
pipeline y realiza la inferencia localmente.

### 15.6 Web app

Opción simple:

```text
doble clic en Abrir web app Cono de Carga.bat
```

Opción manual:

```powershell
cd D:\Memoria\Cono
.\web_app\.venv\Scripts\Activate.ps1
streamlit run web_app\app.py --server.address 127.0.0.1 --server.port 8502
```

## 16. Estado de validación y pruebas

La validación realizada hasta ahora combina evaluación experimental, revisión
visual y una suite automatizada inicial:

- pruebas con máscaras manuales CVAT;
- pruebas Data prueba 1, 2 y 3;
- inspección de overlays;
- procesamiento desde videos locales;
- evaluación balanceada de rendimiento del analítico sobre 90 imágenes;
- matrices de confusión y exactitud por clase;
- repetición del resultado de rendimiento con indicadores geométricos.

En `tests/` existen cuatro pruebas automatizadas versionadas:

- ejecución del adaptador de inferencia con el mismo Python local y sin Docker;
- conservación de observaciones válidas en histogramas;
- interpolación lineal de percentiles;
- exclusión de filas inválidas de los resúmenes estadísticos.

La cobertura sigue siendo parcial. Faltan pruebas unitarias o de integración
para normalización de features, reglas y empates, detección de bordes,
extracción de perfil, adaptación multiclase, veredicto por video y paths
Windows/Unicode.

## 17. Estado actual y decisiones vigentes

### Se considera vigente

- Clases: `centrado`, `izquierda`, `derecha`, `no_evaluable`.
- Umbrales: `peak=0.06`, `area=0.07`.
- Pesos: `0.5/0.5`.
- Máscara principal: `largest`.
- Descarte si toca cualquier borde.
- Mediciones relativas a la máscara de carga.
- Un único modelo entrenado de segmentación, con clase `material=1`, como etapa
  auxiliar del analítico.
- El mismo archivo binario de segmentación se utiliza en Granulometría.
- Rendimiento vigente del analítico geométrico: 88,89 % en 90 imágenes
  balanceadas, con 90/90 evaluables.

### No se considera resuelto

- Detección o segmentación de tolva.
- Control de calidad automático de máscaras incompletas.
- Empate entre pico y área sin sesgo.
- Calibración sobre más cámaras, iluminación y fechas.
- Validación temporal por secuencias independientes.
- Cobertura automatizada completa de geometría, reglas y pipeline.
- Despliegue compartido y versionado formal del modelo de segmentación.

## 18. Próximos pasos recomendados

Prioridad 1:

1. Repetir la evaluación de rendimiento desde el pipeline completo después de
   cambios en la segmentación.
2. Agregar validación automática de los nombres de clases del modelo.
3. Controlar máscaras incompletas que no alcanzan el borde de imagen.

Prioridad 2:

1. Resolver explícitamente empates `0.5/0.5`.
2. Ampliar los tests unitarios de reglas y geometría.
3. Evaluar `imgsz=960` y `1280`.
4. Incorporar más imágenes Camera 7/8, iluminación extrema y bordes de tolva.
5. Revisar consistencia de etiquetas CVAT en laterales y fragmentos aislados.

Prioridad 3:

1. Entrenar o integrar una máscara de tolva.
2. Restringir instancias de material al interior de la tolva.
3. Crear métricas de cobertura y calidad de máscara.
4. Marcar como `no_evaluable` máscaras con subsegmentación evidente.
5. Evaluar veredictos por video contra secuencias etiquetadas completas.

## 19. Advertencias para otra IA

- No interpretar bbox o centroide como información física del camión.
- No asumir que un bbox incompleto es un error del bbox: primero revisar la máscara YOLO.
- No usar `merge_all` indiscriminadamente sin ROI de tolva.
- El modelo de segmentación vigente debe ejecutarse con `class_id=1`
  (`material`); `class_id=0` selecciona `caex`.
- No asumir que los parámetros de inferencia son comunes solo porque Cono y
  Granulometría usan el mismo checkpoint.
- No asumir que mAP de segmentación implica la misma exactitud del analítico.
- No asumir que los paths guardados en CSV siguen válidos después de mover outputs.
- No revelar `.env.mlflow` ni credenciales de Nextcloud.
- No modificar o borrar datos, modelos u outputs sin comprobar su procedencia.
- Mantener los cambios del proyecto dentro de `D:\Memoria\Cono`.

## 20. Índice rápido de archivos

| Archivo | Función |
|---|---|
| `configs/analytic_config.yaml` | Umbrales y parámetros del analítico. |
| `src/cone_load/masks.py` | Conversión, morfología y componente principal. |
| `src/cone_load/profile.py` | Perfil superior, suavizado y picos. |
| `src/cone_load/features.py` | Bbox, centroide, áreas, features y validez. |
| `src/cone_load/rules.py` | Clasificación por reglas. |
| `src/cone_load/visualization.py` | Overlays. |
| `scripts/run_simple_analytic.py` | Ejecución del analítico por imágenes. |
| `scripts/run_pipeline_from_videos.py` | Pipeline completo desde videos locales. |
| `scripts/run_manager_sample_from_images.py` | Procesamiento masivo de imágenes. |
| `scripts/evaluate_technical_test.py` | Evaluación contra carpetas esperadas. |
| `scripts/create_technical_comparisons.py` | Comparaciones visuales lado a lado. |
| `perception/configs/perception_config.yaml` | Datos, entrenamiento, predicción y MLflow. |
| `perception/scripts/prepare_yolo_dataset.py` | CVAT -> YOLO-seg. |
| `perception/scripts/train_yolo_seg.py` | Entrenamiento GPU y MLflow. |
| `perception/scripts/predict_to_analytic_masks.py` | YOLO -> máscara RGB analítica. |
| `perception/docker-compose.yml` | Entorno GPU para entrenamiento y validación. |
| `requirements/runtime.txt` | Dependencias de inferencia, analítico y web app. |
| `tests/test_local_inference.py` | Verificación de inferencia local sin Docker. |
| `tests/test_statistics.py` | Pruebas de resúmenes y distribuciones. |
| `web_app/app.py` | Interfaz Streamlit. |
| `web_app/src/pipeline_runner.py` | Ejecución y streaming de consola. |
| `tools/frame_hunter_nextcloud/main.py` | Descarga y extracción desde Nextcloud. |
| `tools/reporting/generate_report_figures.py` | Regeneración de figuras del informe. |

## 21. Conclusión técnica

El proyecto ya dispone de un pipeline funcional de extremo a extremo y de una
evaluación cuantitativa inicial. La geometría es simple, interpretable y
trazable, pero depende directamente de la calidad de segmentación y de
referencias internas a la máscara. Con la configuración vigente, el analítico
geométrico alcanza 80/90 aciertos, equivalentes a 88,89 %, sobre el conjunto
balanceado de rendimiento.

El único modelo entrenado de segmentación se usa como etapa auxiliar común con
Granulometría. La separación entre proyectos ocurre después de obtener la
máscara: Cono construye descriptores geométricos internos, mientras
Granulometría normaliza la carga y clasifica patches. Mantener explícito este
límite permite compartir la segmentación sin confundir los objetivos,
parámetros y evaluaciones de cada analítico.

Los cambios de mayor valor son fortalecer el control de calidad de máscaras,
resolver explícitamente los casos geométricos ambiguos y ampliar la validación a
secuencias, cámaras y condiciones independientes.
