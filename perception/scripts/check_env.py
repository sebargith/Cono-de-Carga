from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def print_nvidia_smi() -> None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        print("nvidia-smi: no encontrado")
        return
    print(f"nvidia-smi: {exe}")
    try:
        result = subprocess.run(
            [exe],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
    except Exception as exc:
        print(f"nvidia-smi error: {type(exc).__name__}: {exc}")


def main() -> int:
    print(f"python: {sys.version}")
    print(f"executable: {sys.executable}")
    print()
    print_nvidia_smi()
    print()

    for module in ["torch", "torchvision", "ultralytics", "mlflow", "boto3", "cv2", "numpy", "yaml"]:
        print(f"{module}: {'OK' if module_exists(module) else 'NO INSTALADO'}")

    print()
    if module_exists("torch"):
        import torch

        print(f"torch.__version__: {torch.__version__}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        print("torch no esta instalado; aun no se puede entrenar con GPU desde este entorno.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
