@echo off
echo ==========================================
echo   AI Safety Analyst - SIH26165
echo   Oil India Limited
echo ==========================================
echo.

echo [1/2] Starting FastAPI backend on port 8000...
start "Backend" cmd /c "cd /d C:\Users\hp\Desktop\SIF && py -3.12 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo [2/2] Starting React frontend on port 5173...
start "Frontend" cmd /c "cd /d C:\Users\hp\Desktop\SIF\frontend && npm run dev"

echo.
echo ==========================================
echo   Servers starting...
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo ==========================================
echo.
echo Press any key to open browser...
pause >nul
start http://localhost:5173
