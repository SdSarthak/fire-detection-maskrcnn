@echo off
REM Setup script for Mask R-CNN Fire Detection project (Windows)
REM Initializes project structure and installs dependencies

echo ==================================
echo Mask R-CNN Fire Detection - Setup
echo ==================================
echo.

REM Check Python version
echo Step 1: Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Warning: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo.

REM Setup segmentation environment
echo Step 2: Setting up segmentation environment...
cd segmentation

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo Virtual environment created
echo.

REM Install dependencies
echo Step 3: Installing segmentation dependencies...
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo Dependencies installed
echo.

REM Create directories
echo Step 4: Creating project directories...
if not exist data\train mkdir data\train
if not exist data\val mkdir data\val
if not exist data\test mkdir data\test
if not exist data\annotations mkdir data\annotations
if not exist weights mkdir weights
if not exist outputs mkdir outputs
echo Directories created
echo.

cd ..

REM Setup Flask application
echo Step 5: Setting up Flask application...
cd mlops\flask_app

if not exist uploads mkdir uploads
echo Flask upload directory created
echo.

cd ..\..

REM Display setup summary
echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo Project structure:
echo   segmentation/        - Model training and inference
echo   mlops/               - Deployment infrastructure
echo.
echo Next steps:
echo 1. Prepare training data:
echo    - Place images in segmentation\data\train\, val\, test\
echo    - Create annotations using VGG Annotator
echo    - Place JSON files in segmentation\data\annotations\
echo.
echo 2. Train the model:
echo    cd segmentation
echo    python train.py
echo.
echo 3. Run inference:
echo    python infer.py
echo.
echo 4. Deploy to GCP:
echo    Read mlops\DEPLOYMENT_GUIDE.md for detailed instructions
echo.
echo 5. Local Docker testing:
echo    docker build -t fire-detection -f mlops\flask_app\Dockerfile .
echo    docker run -p 5000:5000 fire-detection
echo.
echo Documentation:
echo   - README.md               - Project overview
echo   - segmentation\README.md  - Model training guide
echo   - mlops\README.md         - Deployment guide
echo   - mlops\DEPLOYMENT_GUIDE.md - Complete GCP setup
echo.
pause
