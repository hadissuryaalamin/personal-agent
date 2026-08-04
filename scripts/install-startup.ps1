<#
.SYNOPSIS
  Register personal-agent with Task Scheduler so it starts quietly at every login.

.DESCRIPTION
  Creates the "PersonalAgent" task: At log on trigger, pythonw.exe as the action
  (no console window), "Start in" set to the repo folder.

  Runs without elevation by default, which suits HOTKEY_BACKEND=pynput. Pass
  -Elevated if you switch to HOTKEY_BACKEND=keyboard, which needs highest
  privileges.

  Registering the task itself still requires PowerShell as Administrator.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [string]$TaskName = "PersonalAgent",
    [switch]$Uninstall,
    # Run the task with highest privileges — needed for HOTKEY_BACKEND=keyboard
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
    throw "This script needs PowerShell running as Administrator."
}

# --- Uninstall ---
if ($Uninstall) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        Write-Host "Task '$TaskName' does not exist, nothing to do." -ForegroundColor Yellow
        return
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Task '$TaskName' removed." -ForegroundColor Green
    Write-Host "Stop any running process by hand:  Get-Process pythonw | Stop-Process"
    return
}

# --- Locate pythonw.exe ---
$PythonW = Join-Path $RepoRoot ".venv-agent\Scripts\pythonw.exe"
if (-not (Test-Path $PythonW)) {
    $PythonW = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
}
if (-not (Test-Path $PythonW)) {
    throw "pythonw.exe not found in .venv-agent\Scripts. Create the venv first (see README)."
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

# A background task: it must not be stopped by battery, idle, or a time limit
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Task '$TaskName' already exists, replacing it." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName `
    -Description "personal-agent: push-to-talk voice assistant (background)" `
    -Action $action -Trigger $trigger -Principal $principal -Settings $settings | Out-Null

Write-Host "`nTask '$TaskName' registered (At log on, hidden)." -ForegroundColor Green
Write-Host "Test now without logging out :  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Watch the log                :  Get-Content '$RepoRoot\logs\agent.log' -Tail 20 -Wait"
Write-Host "Stop it                      :  Get-Process pythonw | Stop-Process"
Write-Host "Uninstall                    :  .\scripts\install-startup.ps1 -Uninstall"
