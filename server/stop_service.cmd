@echo off
chcp 65001 >nul
echo 停止 Zh200M-C3 服务 ...

schtasks /end /tn "Zh200M-C3-Service" >nul 2>&1
schtasks /delete /tn "Zh200M-C3-Service" /f >nul 2>&1

powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -and ($_.CommandLine -like '*ws_server.py*' -or $_.CommandLine -like '*127.0.0.1:8765:127.0.0.1:8765*') } | ForEach-Object { Write-Host ('kill ' + $_.ProcessId + ' ' + $_.Name); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo 已停止：本地推理服务、SSH 隧道、开机自启任务。
pause
