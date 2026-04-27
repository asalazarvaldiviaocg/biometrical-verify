@echo off
REM ============================================================
REM  biometrical-verify — local auto-launch script
REM  Runs after Windows reboot to bring up the full stack.
REM ============================================================
setlocal EnableDelayedExpansion

set "PROJECT=C:\Users\alexl\Desktop\AGENTE IA\biometrical-verify"
set "DOCKER_BIN=C:\Program Files\Docker\Docker\resources\bin"
set "DOCKER_EXE=C:\Program Files\Docker\Docker\Docker Desktop.exe"
set "PATH=%DOCKER_BIN%;%PATH%"

echo.
echo ========================================
echo  biometrical-verify local launcher
echo ========================================
echo.

REM --- Start Docker Desktop if not already running ---
tasklist /FI "IMAGENAME eq Docker Desktop.exe" 2>nul | find /I "Docker Desktop.exe" >nul
if errorlevel 1 (
    echo [..] Starting Docker Desktop...
    start "" "%DOCKER_EXE%"
) else (
    echo [ok] Docker Desktop already running
)

REM --- Wait for Docker daemon (up to 5 min) ---
echo [..] Waiting for Docker daemon...
set /a ATTEMPT=0
:waitdocker
"%DOCKER_BIN%\docker.exe" info >nul 2>&1
if not errorlevel 1 goto dockerready
set /a ATTEMPT+=1
if %ATTEMPT% GEQ 60 (
    echo [!!] Docker daemon did not start after 5 minutes.
    echo     Open Docker Desktop manually, accept TOS, skip sign-in.
    pause
    exit /b 1
)
timeout /t 5 /nobreak >nul
goto waitdocker

:dockerready
echo [ok] Docker daemon ready

REM --- Bring up the stack ---
cd /d "%PROJECT%"
echo.
echo [..] Building and starting services (first run takes ~5-10 min)...
"%DOCKER_BIN%\docker.exe" compose up -d --build
if errorlevel 1 (
    echo [!!] docker compose failed. Check: docker compose logs api
    pause
    exit /b 1
)

REM --- Run migrations ---
echo [..] Running database migrations...
"%DOCKER_BIN%\docker.exe" compose run --rm api alembic upgrade head

REM --- Generate dev JWT token and save to desktop ---
echo [..] Generating dev JWT token...
for /f "delims=" %%t in ('"%DOCKER_BIN%\docker.exe" compose run --rm api python scripts/issue_dev_token.py user-dev-1 2^>nul') do set "JWT_TOKEN=%%t"
if defined JWT_TOKEN (
    > "%USERPROFILE%\Desktop\biometrical-verify-TOKEN.txt" (
        echo # biometrical-verify dev JWT token
        echo # Paste this into the frontend's Authorization header as:
        echo # Bearer ^<token^>
        echo.
        echo !JWT_TOKEN!
    )
    echo [ok] Token saved to: %USERPROFILE%\Desktop\biometrical-verify-TOKEN.txt
)

REM --- Start frontend in a new window ---
echo [..] Starting React frontend...
start "biometrical-verify frontend" cmd /k "cd /d ""%PROJECT%\frontend"" && npm install && npm run dev"

REM --- Wait for frontend to be reachable, then open browser ---
echo [..] Waiting for frontend (http://localhost:5173)...
set /a ATTEMPT=0
:waitfe
curl -s -o nul -w "%%{http_code}" http://localhost:5173 2>nul | find "200" >nul
if not errorlevel 1 goto feready
set /a ATTEMPT+=1
if %ATTEMPT% GEQ 60 (
    echo [!!] Frontend did not come up after 5 minutes.
    goto done
)
timeout /t 5 /nobreak >nul
goto waitfe

:feready
echo [ok] Frontend ready
start "" "http://localhost:5173"

:done
echo.
echo ========================================
echo  All services running
echo ========================================
echo  API:      http://localhost:8000/docs
echo  Frontend: http://localhost:5173
echo  MinIO:    http://localhost:9001  (biominio / biominio-secret)
echo  JWT:      %USERPROFILE%\Desktop\biometrical-verify-TOKEN.txt
echo.
echo This window stays open. Close to keep services running.
echo To stop: cd "%PROJECT%" ^&^& docker compose down
echo.
pause
