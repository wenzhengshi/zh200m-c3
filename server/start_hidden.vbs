' Zh200M-C3 autostart: run the guardian hidden
Set sh = CreateObject("WScript.Shell")
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""<PROJECT_DIR>\server\run_service.ps1""", 0, False
