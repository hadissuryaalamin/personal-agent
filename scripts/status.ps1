<#
.SYNOPSIS
  Check whether the agent is running, and how it is doing.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\status.ps1
#>
[CmdletBinding()]
param([string]$TaskName = "PersonalAgent")

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogFile = Join-Path $RepoRoot "logs\agent.log"

function Row($label, $value, $colour = "Gray") {
    Write-Host ("  {0,-12} " -f $label) -NoNewline
    Write-Host $value -ForegroundColor $colour
}

Write-Host "`n=== personal-agent ===" -ForegroundColor Cyan

# --- Process ---
# Both python.exe AND pythonw.exe. Autostart uses pythonw (no console), but a
# manual run from a terminal is python.exe — and while this script only looked
# for pythonw, a manually started agent was invisible to it.
# The consequence was not cosmetic: it reported STOPPED while another agent was
# still grabbing the same hotkey, so every press was caught by both.
$proc = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*src.agent*" }

if ($proc) {
    # The venv python.exe is a stub that runs the real interpreter as a child,
    # so one agent legitimately shows up as two processes. What counts is how
    # many distinct LAUNCHES there are — two separate agents cannot share one.
    $pids = ($proc | ForEach-Object { $_.ProcessId }) -join ", "

    # Processes launched within this many seconds of each other are one agent.
    # Grouping by the exact second was wrong: the launcher and its child can
    # land either side of a second boundary (measured at 11:55:49 and 11:55:50),
    # and the script then reported two agents when there was one. A false alarm
    # here is not harmless — it sends you off killing a process you need.
    $LaunchWindow = 5
    $sorted = @($proc | Sort-Object CreationDate)
    $generations = @()
    $current = @($sorted[0])
    foreach ($p in $sorted | Select-Object -Skip 1) {
        if (($p.CreationDate - $current[0].CreationDate).TotalSeconds -le $LaunchWindow) {
            $current += $p
        } else {
            $generations += , $current
            $current = @($p)
        }
    }
    $generations += , $current
    $started = ($proc | Sort-Object CreationDate | Select-Object -First 1).CreationDate
    $age = [datetime]::Now - $started
    Row "Agent" "RUNNING" "Green"
    Row "PID" $pids

    if ($generations.Count -gt 1) {
        Row "WARNING" "$($generations.Count) AGENTS RUNNING AT ONCE" "Red"
        Write-Host "               They all grab the same hotkey."
        foreach ($g in $generations) {
            $ids = ($g | ForEach-Object { $_.ProcessId }) -join ", "
            Write-Host "               started $($g[0].CreationDate)  PID $ids"
        }
        Write-Host "               Stop the old one:  Stop-Process -Id <PID> -Force" -ForegroundColor Yellow
    }
    $uptime = if ($age.Days -gt 0) {
        "{0}d {1}h" -f $age.Days, $age.Hours
    } elseif ($age.Hours -gt 0) {
        "{0}h {1}m" -f $age.Hours, $age.Minutes
    } else {
        "{0}m" -f $age.Minutes
    }
    Row "Since" ("{0}  (up {1})" -f $started, $uptime)
} else {
    Row "Agent" "STOPPED" "Red"
}

# --- Task Scheduler ---
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Row "Autostart" "installed ($($task.State))" "Green"
} else {
    Row "Autostart" "not installed" "Yellow"
}

# --- Models on the GPU ---
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $vram = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)
    Row "VRAM" $vram
    # Only meaningful while the process is alive, AND only for log lines that
    # belong to THIS process — models die with the process, so a "loaded" note
    # from a previous run no longer applies.
    if ($proc -and (Test-Path $LogFile)) {
        # The pattern follows the messages in stt.py rather than a backend name.
        # It used to read "Whisper ready in", which stopped matching the moment
        # we moved to Parakeet, so the status always claimed "not loaded".
        #
        # "released from memory" has to be specific: the line "Model will be
        # released after 15 min idle" is a startup announcement, not an event.
        $last = Get-Content $LogFile |
            Select-String -Pattern "ready in|released from memory" |
            Where-Object {
                $_.Line -match '^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})' -and
                [datetime]::ParseExact($Matches[1], 'yyyy-MM-dd HH:mm:ss', $null) -ge $started
            } |
            Select-Object -Last 1
        if ($last -match "released from memory") {
            Row "STT model" "released (next question pays a reload)" "Yellow"
        } elseif ($last) {
            Row "STT model" "loaded" "Green"
        } else {
            Row "STT model" "not loaded yet" "Yellow"
        }
    }
}

# --- Last activity ---
if (Test-Path $LogFile) {
    # Session mode writes "User (turn 3):", press mode writes "User:". This used
    # to look only for "User:", so in session mode the timestamp stuck at the
    # last pre-session conversation and it looked like the agent had been idle
    # for hours.
    $lastUser = Get-Content $LogFile | Select-String -Pattern "agent: User[ (:]" | Select-Object -Last 1
    if ($lastUser) {
        $stamp = ($lastUser.Line -split "INFO")[0].Trim()
        Row "Last used" $stamp
    }
}

Write-Host ""
if (-not $proc) {
    Write-Host "Start :  Start-ScheduledTask -TaskName $TaskName" -ForegroundColor Cyan
    Write-Host "   or :  .\.venv-agent\Scripts\python.exe -m src.agent   (in a terminal)" -ForegroundColor Cyan
} else {
    # The stop command MUST match how it was started. This used to always
    # suggest Stop-ScheduledTask, but a manually started agent runs as
    # python.exe and is not managed by that task at all — the command looked
    # like it succeeded while the agent kept running.
    $viaTask = @($proc | Where-Object { $_.Name -eq "pythonw.exe" }).Count -gt 0
    if ($viaTask) {
        Write-Host "Stop  :  Stop-ScheduledTask -TaskName $TaskName" -ForegroundColor Cyan
    } else {
        $main = ($proc | Sort-Object CreationDate | Select-Object -First 1).ProcessId
        Write-Host "Stop  :  Ctrl+C in its terminal, or  Stop-Process -Id $main -Force" -ForegroundColor Cyan
    }
}
Write-Host "Log   :  Get-Content '$LogFile' -Tail 20 -Wait`n" -ForegroundColor Cyan
