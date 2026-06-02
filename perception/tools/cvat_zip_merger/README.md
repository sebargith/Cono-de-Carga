# Combinador de ZIPs CVAT

Usa esta herramienta cuando cada split viene dividido en varios ZIPs, por ejemplo:

```text
Val1.zip
Val2.zip
Train1.zip
Train2.zip
```

Coloca los ZIPs en:

```text
D:\Memoria\Cono\perception\tools\cvat_zip_merger\input\train\
D:\Memoria\Cono\perception\tools\cvat_zip_merger\input\val\
D:\Memoria\Cono\perception\tools\cvat_zip_merger\input\test\
```

Ejecuta:

```powershell
cd D:\Memoria\Cono\perception

python .\tools\cvat_zip_merger\merge_cvat_zips.py --overwrite
```

La salida queda en:

```text
D:\Memoria\Cono\perception\tools\cvat_zip_merger\output\train\
D:\Memoria\Cono\perception\tools\cvat_zip_merger\output\val\
D:\Memoria\Cono\perception\tools\cvat_zip_merger\output\test\
```

Luego copia esas tres carpetas a:

```text
D:\Memoria\Cono\perception\data\raw\
```

o usa PowerShell:

```powershell
Copy-Item .\tools\cvat_zip_merger\output\train .\data\raw\train -Recurse -Force
Copy-Item .\tools\cvat_zip_merger\output\val .\data\raw\val -Recurse -Force
Copy-Item .\tools\cvat_zip_merger\output\test .\data\raw\test -Recurse -Force
```

Cada salida queda como una exportacion CVAT combinada:

```text
train\
  labelmap.txt
  JPEGImages\
  SegmentationClass\
```

Los nombres de archivos se prefijan con el nombre del ZIP para evitar colisiones.

