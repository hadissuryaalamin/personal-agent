<#
.SYNOPSIS
  Cek agent lagi nyala atau nggak, plus kondisinya.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\status.ps1
#>
[CmdletBinding()]
param([string]$TaskName = "PersonalAgent")

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogFile = Join-Path $RepoRoot "logs\agent.log"

function Baris($label, $nilai, $warna = "Gray") {
    Write-Host ("  {0,-12} " -f $label) -NoNewline
    Write-Host $nilai -ForegroundColor $warna
}

Write-Host "`n=== personal-agent ===" -ForegroundColor Cyan

# --- Proses ---
$proc = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*agent.main*" }

if ($proc) {
    # venv pythonw.exe itu stub yang jalanin interpreter aslinya sebagai anak,
    # jadi satu agent wajar muncul sebagai dua proses
    $pids = ($proc | ForEach-Object { $_.ProcessId }) -join ", "
    $mulai = ($proc | Sort-Object CreationDate | Select-Object -First 1).CreationDate
    $lama = [datetime]::Now - $mulai
    Baris "Agent" "NYALA" "Green"
    Baris "PID" $pids
    $durasi = if ($lama.Days -gt 0) {
        "{0} hari {1} jam" -f $lama.Days, $lama.Hours
    } elseif ($lama.Hours -gt 0) {
        "{0} jam {1} menit" -f $lama.Hours, $lama.Minutes
    } else {
        "{0} menit" -f $lama.Minutes
    }
    Baris "Sejak" ("{0}  (jalan {1})" -f $mulai, $durasi)
} else {
    Baris "Agent" "MATI" "Red"
}

# --- Task Scheduler ---
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Baris "Autostart" "terpasang ($($task.State))" "Green"
} else {
    Baris "Autostart" "nggak terpasang" "Yellow"
}

# --- Model di GPU ---
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $vram = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)
    Baris "VRAM" $vram
    # Cuma relevan kalau prosesnya hidup, DAN baris lognya milik proses ini —
    # model ikut mati bareng prosesnya, jadi catatan "dimuat" dari proses
    # sebelumnya nggak berlaku lagi.
    if ($proc -and (Test-Path $LogFile)) {
        $last = Get-Content $LogFile |
            Select-String -Pattern "Whisper siap dalam|Whisper dilepas" |
            Where-Object {
                $_.Line -match '^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})' -and
                [datetime]::ParseExact($Matches[1], 'yyyy-MM-dd HH:mm:ss', $null) -ge $mulai
            } |
            Select-Object -Last 1
        if ($last -match "dilepas") {
            Baris "Model STT" "dilepas (pertanyaan berikutnya kena muat ulang)" "Yellow"
        } elseif ($last) {
            Baris "Model STT" "dimuat" "Green"
        } else {
            Baris "Model STT" "belum dimuat" "Yellow"
        }
    }
}

# --- Memori ---
$facts = Join-Path $RepoRoot "memory\facts.md"
$hist = Join-Path $RepoRoot "memory\history.json"
$nFakta = if (Test-Path $facts) { (Get-Content $facts | Where-Object { $_.Trim() }).Count } else { 0 }
$nPesan = if (Test-Path $hist) {
    try { ((Get-Content $hist -Raw | ConvertFrom-Json).messages).Count } catch { "?" }
} else { 0 }
Baris "Memori" "$nFakta fakta, $nPesan pesan"

# --- Aktivitas terakhir ---
if (Test-Path $LogFile) {
    $terakhir = Get-Content $LogFile | Select-String -Pattern "User:" | Select-Object -Last 1
    if ($terakhir) {
        $t = ($terakhir.Line -split "INFO")[0].Trim()
        Baris "Terakhir" $t
    }
}

Write-Host ""
if (-not $proc) {
    Write-Host "Nyalain :  Start-ScheduledTask -TaskName $TaskName" -ForegroundColor Cyan
} else {
    Write-Host "Matiin  :  Stop-ScheduledTask -TaskName $TaskName" -ForegroundColor Cyan
}
Write-Host "Log     :  Get-Content '$LogFile' -Tail 20 -Wait`n" -ForegroundColor Cyan
