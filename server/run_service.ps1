# Zh200M-C3 服务守护脚本
# 职责：保持「本地推理服务」+「到中转服务器的 SSH 反向隧道」一直活着。
# 任一进程退出后 10 秒内自动重启。单实例运行，重复启动会直接退出。

$ErrorActionPreference = 'Continue'

$serverDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $serverDir
$logDir = Join-Path $serverDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir 'service.log'

# ---- 单实例锁：开机自启 + 手动启动同时发生时不产生两份守护 ----
$created = $false
$mutex = New-Object System.Threading.Mutex($true, 'Local\Zh200M-C3-Guardian', [ref]$created)
if (-not $created) { exit 0 }

function Write-Log([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $logFile -Value $line -Encoding UTF8
}

# 日志超过 5MB 就轮转一次，避免长期运行把磁盘写满
if ((Test-Path $logFile) -and ((Get-Item $logFile).Length -gt 5MB)) {
    Move-Item $logFile "$logFile.1" -Force -ErrorAction SilentlyContinue
}

$pythonCandidates = @(
    'C:\Python313\python.exe'  # 按需改成你的路径
)
$python = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) { $python = (Get-Command python -ErrorAction SilentlyContinue).Source }

$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
if (-not (Test-Path $ssh)) { $ssh = (Get-Command ssh).Source }

$remote = 'root@YOUR_SERVER_IP'

# 清理上次遗留的孤儿进程，否则新进程会因为端口被占而永远起不来
foreach ($conn in (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue)) {
    $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
    if ($proc -and $proc.ProcessName -like 'python*') {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Write-Log "清理占用 8765 的旧进程 pid=$($proc.Id)"
        Start-Sleep -Seconds 1
    }
}

Write-Log "守护启动  python=$python  ssh=$ssh"

$srv = $null
$tun = $null
$lastState = ''

while ($true) {
    if ($null -eq $srv -or $srv.HasExited) {
        $srv = Start-Process -FilePath $python `
            -ArgumentList @('-u', (Join-Path $serverDir 'ws_server.py'), '--port', '8765', '--threads', '4') `
            -WorkingDirectory $root -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logDir 'server.out.log') `
            -RedirectStandardError  (Join-Path $logDir 'server.err.log') `
            -PassThru
        Write-Log "启动推理服务 pid=$($srv.Id)"
    }

    if ($null -eq $tun -or $tun.HasExited) {
        $tun = Start-Process -FilePath $ssh `
            -ArgumentList @('-N', '-R', '127.0.0.1:8765:127.0.0.1:8765',
                            '-o', 'ExitOnForwardFailure=yes',
                            '-o', 'ServerAliveInterval=20',
                            '-o', 'ServerAliveCountMax=3',
                            '-o', 'TCPKeepAlive=yes',
                            '-o', 'BatchMode=yes',
                            '-o', 'StrictHostKeyChecking=accept-new',
                            $remote) `
            -WindowStyle Hidden `
            -RedirectStandardError (Join-Path $logDir 'tunnel.err.log') `
            -PassThru
        Write-Log "启动 SSH 隧道 pid=$($tun.Id)"
    }

    $state = "server=$($srv.Id)/$(if($srv.HasExited){'down'}else{'up'}) tunnel=$($tun.Id)/$(if($tun.HasExited){'down'}else{'up'})"
    if ($state -ne $lastState) { Write-Log $state; $lastState = $state }

    Start-Sleep -Seconds 10
}
