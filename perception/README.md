# Modelo de percepcion para carga visible en CAEX

Este repositorio prepara datos exportados desde CVAT y entrena un modelo YOLO de segmentacion para detectar automaticamente la carga visible del camion.

La entrada esperada es la exportacion CVAT tipo **Segmentation Mask**:

```text
labelmap.txt
JPEGImages/
SegmentationClass/
```

La clase usada es:

```text
Carga:102,255,102::
background:0,0,0::
```

El pipeline convierte esas mascaras RGB a formato YOLO-seg y entrena un modelo desde una arquitectura `.yaml`, no desde un `best.pt` ya entrenado.

## 1. Docker Con GPU

Esta es la forma recomendada. Docker encapsula Python, PyTorch, CUDA runtime, Ultralytics y dependencias. No uses `.venv-yolo` si vas a entrenar con Docker.

Requisitos:

- Docker Desktop instalado.
- Driver NVIDIA instalado.
- GPU visible desde PowerShell con `nvidia-smi`.
- Docker Desktop con backend WSL2 y soporte GPU.

Verifica GPU en el host:

```powershell
nvidia-smi
```

Construye la imagen:

```powershell
cd D:\Memoria\Cono\perception
docker compose build
```

Construir la imagen es necesario la primera vez, o cuando cambies:

- `Dockerfile`
- `requirements/docker.txt`
- Versiones base de CUDA/PyTorch/Ultralytics

No hace falta reconstruir la imagen cada vez que cambies:

- `configs/perception_config.yaml`
- scripts en `scripts/`
- codigo en `src/`
- datos en `data/raw/`

Esas carpetas se montan como volumen dentro del contenedor.

Levanta el contenedor:

```powershell
docker compose up -d
```

Verifica dependencias dentro del contenedor:

```powershell
docker compose exec cone-yolo python scripts/check_env.py
```

Debe verse:

```text
torch: OK
torchvision: OK
ultralytics: OK
mlflow: OK
boto3: OK
torch.cuda.is_available(): True
```

Abre una shell si necesitas inspeccionar:

```powershell
docker compose exec cone-yolo bash
```

Apaga el contenedor:

```powershell
docker compose down
```

## 1.1 Correr Mas De Un Experimento

Flujo recomendado:

```powershell
cd D:\Memoria\Cono\perception

# Solo primera vez o cuando cambien Dockerfile/requirements
docker compose build

# Mantener el contenedor vivo para correr varios experimentos
docker compose up -d

# Verificar una vez
docker compose exec cone-yolo python scripts/check_env.py
```

Luego puedes correr muchos experimentos cambiando `--run-name`, `--model`, `--epochs`, `--batch`, `--imgsz` o la configuracion:

```powershell
docker compose exec cone-yolo python scripts/train_yolo_seg.py `
  --config "configs/perception_config.yaml" `
  --model yolov8n-seg.yaml `
  --no-pretrained `
  --epochs 50 `
  --batch 8 `
  --imgsz 640 `
  --device 0 `
  --run-name exp_yolov8n_50e

docker compose exec cone-yolo python scripts/train_yolo_seg.py `
  --config "configs/perception_config.yaml" `
  --model yolov8m-seg.yaml `
  --no-pretrained `
  --epochs 100 `
  --batch 8 `
  --imgsz 640 `
  --device 0 `
  --run-name exp_yolov8m_100e
```

Cuando termines:

```powershell
docker compose down
```

## 2. MLflow

Cada entrenamiento queda registrado en MLflow. Por defecto se usa un tracking local dentro del proyecto:

```text
D:\Memoria\Cono\perception\mlruns\
```

El script registra:

- parametros de entrenamiento;
- configuracion usada;
- `dataset.yaml` y `manifest.csv`;
- metricas finales y mejores metricas leidas desde `runs/train/<run_name>/results.csv`;
- `best.pt` y `last.pt` como artefactos.

Para abrir la interfaz de MLflow:

```powershell
cd D:\Memoria\Cono\perception
docker compose --profile ui up -d mlflow-ui
```

Luego abre:

```text
http://localhost:5000
```

Si quieres usar un servidor remoto de MLflow, copia `.env.mlflow.example` como `D:\Memoria\Cono\perception\.env.mlflow` y completa los valores:

```text
MLFLOW_TRACKING_URI=http://servidor-mlflow:5000
MLFLOW_S3_ENDPOINT_URL=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

`docker-compose.yml` carga ese archivo con `env_file`. Si no defines `MLFLOW_TRACKING_URI`, se usa:

```text
file:/workspace/mlruns
```

Si tu servidor MLflow guarda artefactos en S3 o MinIO, el contenedor necesita `boto3`. Ya esta incluido en `requirements/docker.txt`. Si agregaste esta dependencia despues de construir la imagen, reconstruye:

```powershell
docker compose down
docker compose build --no-cache
docker compose up -d
```

La configuracion MLflow esta en:

```yaml
mlflow:
  enabled: true
  tracking_uri: file:/workspace/mlruns
  experiment_name: cono_carga_yolo_seg
  log_run_dir_artifacts: false
  log_weight_artifacts: true
```

## 3. Estructura Encapsulada

Todo queda dentro de `D:\Memoria\Cono\perception`:

```text
data/raw/        exportaciones CVAT de entrada
data/yolo/       dataset YOLO generado
runs/            corridas de train/val/predict
models/primary/  modelo principal usado por el pipeline
models/alternative/ checkpoint alternativo versionado
models/<run>/    pesos generados por nuevos entrenamientos
mlruns/          tracking local de MLflow
.cache/          cache de Ultralytics/Matplotlib
```

Copia o descomprime tus exportaciones CVAT separadas por split en:

```text
D:\Memoria\Cono\perception\data\raw\train\
D:\Memoria\Cono\perception\data\raw\val\
D:\Memoria\Cono\perception\data\raw\test\
```

Cada carpeta debe ser una exportacion CVAT completa:

```text
train/
  labelmap.txt
  JPEGImages/
  SegmentationClass/

val/
  labelmap.txt
  JPEGImages/
  SegmentationClass/

test/
  labelmap.txt
  JPEGImages/
  SegmentationClass/
```

Si cada split viene dividido en varios ZIPs, usa primero:

```text
D:\Memoria\Cono\perception\tools\cvat_zip_merger\
```

Instrucciones:

```text
D:\Memoria\Cono\perception\tools\cvat_zip_merger\README.md
```

## 4. Preparar Dataset YOLO

Si tus datos ya estan separados en `train/val/test`, no pases rutas. El script detecta automaticamente `data/raw/train`, `data/raw/val` y `data/raw/test` y respeta esa separacion:

```powershell
cd D:\Memoria\Cono\perception

docker compose exec cone-yolo python scripts/prepare_yolo_dataset.py --overwrite
```

Equivalente explicito:

```powershell
docker compose exec cone-yolo python scripts/prepare_yolo_dataset.py `
  --train-cvat-dir "data/raw/train" `
  --val-cvat-dir "data/raw/val" `
  --test-cvat-dir "data/raw/test" `
  --config "configs/perception_config.yaml" `
  --overwrite
```

También se pueden pasar una o más exportaciones explícitas y generar un split
aleatorio:

```powershell
docker compose exec cone-yolo python scripts/prepare_yolo_dataset.py `
  --cvat-dir "<ruta_exportacion_cvat_1>" `
  --cvat-dir "<ruta_exportacion_cvat_2>" `
  --config "configs/perception_config.yaml" `
  --overwrite
```

Se genera:

```text
D:\Memoria\Cono\perception\data\yolo\carga_visible_seg\dataset.yaml
D:\Memoria\Cono\perception\data\yolo\carga_visible_seg\manifest.csv
```

## 5. Entrenar Desde Cero

El camino principal es entrenar el modelo desde arquitectura YOLO, sin pesos previos:

```powershell
docker compose exec cone-yolo python scripts/train_yolo_seg.py `
  --config "configs/perception_config.yaml" `
  --model yolov8n-seg.yaml `
  --no-pretrained `
  --epochs 50 `
  --batch 8 `
  --imgsz 640 `
  --device 0
```

Modelo mas grande:

```powershell
docker compose exec cone-yolo python scripts/train_yolo_seg.py `
  --config "configs/perception_config.yaml" `
  --model yolov8m-seg.yaml `
  --no-pretrained `
  --epochs 100 `
  --batch 8 `
  --imgsz 640 `
  --device 0
```

Para YOLOv26, copia la arquitectura compatible dentro del repo, por ejemplo:

```text
D:\Memoria\Cono\perception\configs\models\yolo26-seg.yaml
```

y ejecuta:

```powershell
docker compose exec cone-yolo python scripts/train_yolo_seg.py `
  --config "configs/perception_config.yaml" `
  --model "configs/models/yolo26-seg.yaml" `
  --no-pretrained `
  --epochs 100 `
  --batch 8 `
  --imgsz 640 `
  --device 0
```

El uso de `.pt` queda reservado para fine-tuning explicito:

```powershell
docker compose exec cone-yolo python scripts/train_yolo_seg.py `
  --config "configs/perception_config.yaml" `
  --model yolov8n-seg.pt `
  --pretrained `
  --epochs 50 `
  --batch 8 `
  --imgsz 640 `
  --device 0
```

Si `--no-pretrained` se usa junto a un `.pt`, el script falla para evitar confusiones.

## 6. Salidas

Ultralytics guarda el run en:

```text
D:\Memoria\Cono\perception\runs\train\<run_name>\
```

El script copia los pesos principales a:

```text
D:\Memoria\Cono\perception\models\<run_name>\best.pt
D:\Memoria\Cono\perception\models\<run_name>\last.pt
```

## 7. Validar

```powershell
docker compose exec cone-yolo python scripts/validate_yolo_seg.py `
  --model "models/<run_name>/best.pt" `
  --config "configs/perception_config.yaml" `
  --device 0
```

## 8. Inferencia

```powershell
docker compose exec cone-yolo python scripts/predict_yolo_seg.py `
  --model "models/primary/psinet_carga_visible.pt" `
  --source "model_testing/reference/prueba_01/sources" `
  --config "configs/perception_config.yaml" `
  --conf 0.35 `
  --device 0
```

Las predicciones quedan en:

```text
D:\Memoria\Cono\perception\model_testing\outputs\
```

## 8.1 Generar Mascaras Para El Analitico

Para conectar YOLO-seg con el analitico geometrico, usa el adaptador:

```powershell
docker compose exec cone-yolo python scripts/predict_to_analytic_masks.py `
  --model "models/primary/psinet_carga_visible.pt" `
  --images-dir "model_testing/reference/prueba_01/sources" `
  --masks-dir "model_testing/analytic_masks/prueba_01" `
  --conf 0.35 `
  --device 0 `
  --overwrite
```

La configuración selecciona `class_id=1`, correspondiente a `material` en el
modelo principal PSINet.

Este script guarda una mascara `.png` por frame con el mismo nombre base:

```text
model_testing/analytic_masks/prueba_01/
  Camera_8_..._frame000000.png
  manifest.csv
  summary.json
```

Las mascaras se escriben en el mismo formato RGB que espera el analitico:

```text
Carga:     [102, 255, 102]
background:[0, 0, 0]
```

No guardes estas mascaras como JPG/JPEG, porque la compresion puede alterar el color exacto de la clase y el analitico podria no detectarla.

Por defecto, si YOLO detecta mas de una instancia de carga en el frame, se conserva la de mayor area (`analytic_mask_mode: largest`). Esto evita que detecciones secundarias afecten la forma visible de la carga. Si necesitas unir todas las instancias detectadas, usa:

```powershell
docker compose exec cone-yolo python scripts/predict_to_analytic_masks.py `
  --model "models/alternative/best.pt" `
  --class-id 0 `
  --images-dir "<carpeta_de_frames>" `
  --masks-dir "model_testing/analytic_masks/<nombre_prueba>" `
  --mask-mode merge_all `
  --overwrite
```

Luego corre el analitico desde la raiz del proyecto Cono:

```powershell
cd D:\Memoria\Cono

python scripts/run_simple_analytic.py `
  --images-dir "perception\model_testing\reference\prueba_01\sources" `
  --masks-dir "perception\model_testing\analytic_masks\prueba_01" `
  --output-dir "outputs\pipeline_yolo_masks_prueba_01" `
  --config "configs\analytic_config.yaml"
```

El resultado queda en:

```text
D:\Memoria\Cono\outputs\pipeline_yolo_masks_prueba_01\tables\features.csv
D:\Memoria\Cono\outputs\pipeline_yolo_masks_prueba_01\overlays\
```

## 9. Modelos Versionados

El repositorio conserva dos pesos con funciones distintas:

```text
D:\Memoria\Cono\perception\models\primary\psinet_carga_visible.pt
D:\Memoria\Cono\perception\models\alternative\best.pt
```

El primero es el segmentador operativo y usa la clase `material`
(`class_id=1`). El segundo es el checkpoint entrenado en este repositorio y usa
la clase `Carga` (`class_id=0`). También se conserva un caso de inferencia
reproducible con 24 imágenes de entrada y sus resultados:

```text
D:\Memoria\Cono\perception\model_testing\reference\prueba_01\
```

Los modelos, datasets y outputs generados por experimentos futuros permanecen fuera de Git. MLflow mantiene la trazabilidad de las nuevas corridas.

## 10. Parametros

Edita:

```text
D:\Memoria\Cono\perception\configs\perception_config.yaml
```

Parametros importantes:

- `dataset.class_rgb`: color RGB de la clase `Carga` en CVAT.
- `dataset.keep_largest_component`: conserva solo el componente principal de carga.
- `dataset.polygon_epsilon_fraction`: simplificacion del contorno para YOLO-seg.
- `split.train`, `split.val`, `split.test`: particion aleatoria del dataset.
- `training.model`: arquitectura a entrenar, por ejemplo `yolov8n-seg.yaml`.
- `training.pretrained`: `false` para entrenar desde cero.
- `training.epochs`, `batch`, `imgsz`, `device`: hiperparametros principales.
- `prediction.analytic_masks_dir`: carpeta por defecto para mascaras YOLO compatibles con el analitico.
- `prediction.analytic_mask_mode`: `largest` para conservar la instancia principal o `merge_all` para unir instancias.
- `mlflow.enabled`: activa/desactiva tracking.
- `mlflow.tracking_uri`: destino MLflow local o remoto.

## 11. Conexion Con El Analitico Geometrico

El modelo de percepcion produce mascaras automaticas de la carga visible mediante `scripts/predict_to_analytic_masks.py`. Esas mascaras pueden reemplazar las mascaras manuales de CVAT y alimentar el analitico geometrico en `src/cone_load`.
