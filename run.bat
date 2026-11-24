@echo off
cd /d "%~dp0"
if not exist .venv (echo Run setup.bat first. & pause & exit /b 1)
where ffmpeg >nul 2>&1 || (
  if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.1-full_build\bin" (
    set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.1-full_build\bin;%PATH%"
  )
)
echo Starting Clipsmith at http://127.0.0.1:8765  (close this window to stop)
start "" http://127.0.0.1:8765
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8765
pause
