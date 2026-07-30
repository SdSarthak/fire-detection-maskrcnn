@echo off
REM Set up a local development environment for the fire detection project.
setlocal

cd /d "%~dp0"

echo ==========================================
echo  Fire detection with Mask R-CNN - setup
echo ==========================================

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.9+ was not found on PATH.
    exit /b 1
)
python --version

echo.
echo 1/4 Creating the virtual environment...
if not exist venv (
    python -m venv venv
    if errorlevel 1 exit /b 1
)
call venv\Scripts\activate.bat
echo   venv ready

echo.
echo 2/4 Installing dependencies...
python -m pip install --quiet --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1
python -m pip install --quiet -r requirements-dev.txt
if errorlevel 1 exit /b 1
echo   dependencies installed

echo.
echo 3/4 Creating project directories...
for %%D in (
    segmentation\data\train
    segmentation\data\val
    segmentation\data\test
    segmentation\data\annotations
    segmentation\weights
    segmentation\outputs
) do if not exist %%D mkdir %%D
echo   directories ready

echo.
echo 4/4 Preparing configuration...
if not exist .env (
    copy /y .env.example .env >nul
    echo   .env created from .env.example -- edit it before deploying
) else (
    echo   .env already exists, left untouched
)

echo.
echo Setup complete.
echo.
echo Next steps
echo   1. Add images to segmentation\data\{train,val,test}\ and export VIA
echo      annotations to segmentation\data\annotations\.
echo   2. Train:   cd segmentation ^&^& python train.py --epochs 30
echo   3. Infer:   cd segmentation ^&^& python infer.py --input data\test
echo   4. Serve:   set MODEL_PATH=segmentation\weights\fire_detection_model.pt
echo               python mlops\flask_app\app.py
echo   5. Test:    pytest
echo   6. Deploy:  see mlops\DEPLOYMENT_GUIDE.md
echo.

endlocal
