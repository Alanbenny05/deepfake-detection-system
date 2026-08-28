@echo off
echo =======================================================
echo   Deepfake Detection System - Setup & Initialization
echo =======================================================
echo.

:: 1. Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python is not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)
echo [+] Python found.

:: 2. Environment file
echo.
echo [2/6] Setting up environment variables (.env)...
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo [+] Created .env from .env.example
    ) else (
        echo SECRET_KEY=django-insecure-local-dev-key > .env
        echo DEBUG=True >> .env
        echo ALLOWED_HOSTS=* >> .env
        echo [+] Created default .env
    )
) else (
    echo [+] .env file already exists.
)

:: 3. Install requirements
echo.
echo [3/6] Installing Python dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [!] Warning: Some dependencies failed to install. Continuing with basic setup...
)

:: 4. Run Migrations
echo.
echo [4/6] Running database migrations...
python manage.py makemigrations
python manage.py migrate

:: 5. Create Dataset Structure
echo.
echo [5/6] Setting up dataset and sample assets...
python generate_sample_dataset.py

:: 6. Verification Check
echo.
echo [6/6] Checking Django configuration...
python manage.py check

echo.
echo =======================================================
echo   Setup Complete!
echo   To start the development server, run:
echo   python manage.py runserver
echo =======================================================
echo.
pause
