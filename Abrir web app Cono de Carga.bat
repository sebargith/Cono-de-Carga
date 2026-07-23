@echo off
setlocal
chcp 65001 >nul

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"
set "APP_HOST=127.0.0.1"
set "APP_PORT=8502"
set "APP_URL=http://%APP_HOST%:%APP_PORT%"

echo ==========================================
echo Cono de Carga - Web App
echo ==========================================
echo Proyecto: %ROOT%
echo.

if not exist "web_app\app.py" (
  echo ERROR: no se encontro web_app\app.py
  echo Ejecuta este archivo desde la raiz del proyecto Cono.
  pause
  exit /b 1
)

if not exist "web_app\.venv\Scripts\python.exe" (
  echo No existe web_app\.venv. Creando entorno local...
  py -3.12 -m venv "web_app\.venv"
  if errorlevel 1 (
    echo No se pudo crear con py -3.12. Intentando con python...
    python -m venv "web_app\.venv"
  )
  if errorlevel 1 (
    echo ERROR: no se pudo crear el entorno virtual.
    pause
    exit /b 1
  )
)

set "PYTHON=web_app\.venv\Scripts\python.exe"

echo Verificando dependencias de la web app...
"%PYTHON%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
  echo Instalando dependencias en web_app\.venv...
  "%PYTHON%" -m pip install --upgrade pip
  if errorlevel 1 (
    echo ERROR: fallo la actualizacion de pip.
    pause
    exit /b 1
  )
  "%PYTHON%" -m pip install -r "web_app\requirements.txt"
  if errorlevel 1 (
    echo ERROR: fallo la instalacion de dependencias.
    pause
    exit /b 1
  )
)

where docker >nul 2>nul
if errorlevel 1 (
  echo Advertencia: Docker no esta disponible en PATH.
  echo La web app abrira, pero el pipeline con YOLO necesitara Docker levantado.
) else (
  echo Levantando contenedor YOLO si no esta activo...
  pushd "perception"
  docker compose up -d
  if errorlevel 1 (
    echo Advertencia: no se pudo levantar Docker automaticamente.
    echo Puedes abrir la web app igual, pero revisa Docker antes de ejecutar el pipeline.
  )
  popd
)

echo.
echo Abriendo web app en el navegador...
echo Si el navegador no se abre solo, entra a:
echo %APP_URL%
echo.

"%PYTHON%" -m streamlit run "web_app\app.py" ^
  --server.address "%APP_HOST%" ^
  --server.port "%APP_PORT%" ^
  --server.headless false ^
  --browser.gatherUsageStats false

echo.
echo La web app se cerro o Streamlit termino.
pause
