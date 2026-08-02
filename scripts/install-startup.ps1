<#
.SYNOPSIS
  Daftarin personal-agent ke Task Scheduler biar jalan diam-diam tiap login.

.DESCRIPTION
  Bikin task "PersonalAgent": trigger At log on, action pythonw.exe (tanpa console
  window), "Start in" = folder repo.

  Default jalan tanpa elevasi (cocok buat HOTKEY_BACKEND=pynput). Pakai -Elevated
  kalau pindah ke HOTKEY_BACKEND=keyboard, yang butuh highest privileges.

  Registrasi task-nya sendiri tetep butuh PowerShell sebagai Administrator.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [string]$TaskName = "PersonalAgent",
    [switch]$Uninstall,
    # Jalanin task with highest privileges — perlu kalau HOTKEY_BACKEND=keyboard
    [switch]$Elevated
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    throw "Script ini butuh PowerShell yang dijalanin sebagai Administrator."
}

# --- Uninstall ---
if ($Uninstall) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        Write-Host "Task '$TaskName' nggak ada, nggak ngapa-ngapain." -ForegroundColor Yellow
        return
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Task '$TaskName' dihapus." -ForegroundColor Green
    Write-Host "Proses yang lagi jalan matiin manual:  Get-Process pythonw | Stop-Process"
    return
}

# --- Cari pythonw.exe ---
$PythonW = Join-Path $RepoRoot ".venv-agent\Scripts\pythonw.exe"
if (-not (Test-Path $PythonW)) {
    $PythonW = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
}
if (-not (Test-Path $PythonW)) {
    throw "pythonw.exe nggak ketemu di .venv-agent\Scripts. Bikin venv-nya dulu (lihat README)."
}

Write-Host "=== install-startup ===" -ForegroundColor Cyan
Write-Host "Repo   : $RepoRoot"
Write-Host "Python : $PythonW"

$action = New-ScheduledTaskAction -Execute $PythonW -Argument "-m agent.main" -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel $(if ($Elevated) { "Highest" } else { "Limited" })

# Task background: nggak usah dimatiin gara-gara baterai / idle / timeout
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Task '$TaskName' udah ada, ditimpa." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName `
    -Description "personal-agent: voice assistant push-to-talk (background)" `
    -Action $action -Trigger $trigger -Principal $principal -Settings $settings | Out-Null

Write-Host "`nTask '$TaskName' terdaftar (At log on, hidden)." -ForegroundColor Green
Write-Host "Tes sekarang tanpa logout :  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Cek log                   :  Get-Content '$RepoRoot\logs\agent.log' -Tail 20 -Wait"
Write-Host "Matiin                    :  Get-Process pythonw | Stop-Process"
Write-Host "Uninstall                 :  .\scripts\install-startup.ps1 -Uninstall"
