@echo off
chcp 65001 >nul
setlocal
set TASK=Zh200M-C3-Service
set PSCMD=powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0run_service.ps1"

echo 注册开机自启任务 %TASK% ...
schtasks /create /tn "%TASK%" /tr "%PSCMD%" /sc onlogon /rl limited /f
if errorlevel 1 goto fail

echo 立即启动 ...
schtasks /run /tn "%TASK%"
if errorlevel 1 goto fail

echo.
echo 已启动。等待约 12 秒后访问 https://zh200m.xmkjsoft.com/
timeout /t 12 >nul
start "" https://zh200m.xmkjsoft.com/
goto :eof

:fail
echo.
echo 注册或启动失败。请用管理员身份重跑本文件，或直接运行 run_service.ps1。
pause
