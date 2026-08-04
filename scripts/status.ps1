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
# python.exe DAN pythonw.exe. Autostart pakai pythonw (tanpa console), tapi
# kalau dijalanin manual dari terminal prosesnya python.exe — dan waktu skrip
# ini cuma nyari pythonw, agent yang jalan manual nggak pernah kelihatan.
# Akibatnya fatal: skripnya bilang MATI padahal ada agent lain yang ikut
# nyambar hotkey, dan pencetanmu ditangkep dua-duanya.
$proc = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*agent.main*" }

if ($proc) {
    # venv python.exe itu stub yang jalanin interpreter aslinya sebagai anak,
    # jadi satu agent wajar muncul sebagai dua proses. Yang dihitung: berapa
    # WAKTU MULAI yang beda — dua agent terpisah pasti beda detik mulainya.
    $pids = ($proc | ForEach-Object { $_.ProcessId }) -join ", "
    # @() wajib. Tanpa itu, satu grup berisi 2 proses bikin .Count baca jumlah
    # ANGGOTA (2), bukan jumlah grup (1) — dan skripnya teriak alarm palsu.
    $angkatan = @($proc | Group-Object { $_.CreationDate.ToString("s") })
    $mulai = ($proc | Sort-Object CreationDate | Select-Object -First 1).CreationDate
    $lama = [datetime]::Now - $mulai
    Baris "Agent" "NYALA" "Green"
    Baris "PID" $pids

    if ($angkatan.Count -gt 1) {
        Baris "PERINGATAN" "$($angkatan.Count) AGENT JALAN BARENG" "Red"
        Write-Host "               Semuanya nyambar hotkey yang sama."
        foreach ($a in ($angkatan | Sort-Object Name)) {
            $ids = ($a.Group | ForEach-Object { $_.ProcessId }) -join ", "
            Write-Host "               mulai $($a.Group[0].CreationDate)  PID $ids"
        }
        Write-Host "               Matiin yang lama:  Stop-Process -Id <PID> -Force" -ForegroundColor Yellow
    }
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
        # Polanya ngikutin pesan di stt.py, bukan nama backend — dulu ketulis
        # "Whisper siap dalam" dan berhenti cocok begitu pindah ke Parakeet,
        # jadi statusnya selalu bilang "belum dimuat".
        #
        # "dilepas dari memori" harus spesifik: baris "Model bakal dilepas
        # kalau nganggur 15 menit" itu pengumuman setelan pas start, bukan
        # kejadian model dilepas.
        $last = Get-Content $LogFile |
            Select-String -Pattern "siap dalam|dilepas dari memori" |
            Where-Object {
                $_.Line -match '^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})' -and
                [datetime]::ParseExact($Matches[1], 'yyyy-MM-dd HH:mm:ss', $null) -ge $mulai
            } |
            Select-Object -Last 1
        if ($last -match "dilepas dari memori") {
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
    # Mode sesi nulis "User (giliran 3):", mode pencet nulis "User:". Dulu
    # cuma nyari "User:" — jadi di mode sesi jamnya nyangkut di percakapan
    # terakhir sebelum mode sesi dipakai, dan kelihatan kayak agent nggak
    # dipakai berjam-jam.
    $terakhir = Get-Content $LogFile | Select-String -Pattern "agent: User[ (:]" | Select-Object -Last 1
    if ($terakhir) {
        $t = ($terakhir.Line -split "INFO")[0].Trim()
        Baris "Terakhir" $t
    }
}

Write-Host ""
if (-not $proc) {
    Write-Host "Nyalain :  Start-ScheduledTask -TaskName $TaskName" -ForegroundColor Cyan
    Write-Host "   atau :  .\.venv-agent\Scripts\python.exe -m agent.main   (di terminal)" -ForegroundColor Cyan
} else {
    # Perintah matiinnya HARUS ngikutin cara dia dijalanin. Dulu selalu
    # nyaranin Stop-ScheduledTask, padahal agent yang dijalanin manual jalan
    # sebagai python.exe dan sama sekali nggak dikelola task itu — perintahnya
    # kelihatan sukses tapi agent-nya tetep hidup.
    $lewatTask = @($proc | Where-Object { $_.Name -eq "pythonw.exe" }).Count -gt 0
    if ($lewatTask) {
        Write-Host "Matiin  :  Stop-ScheduledTask -TaskName $TaskName" -ForegroundColor Cyan
    } else {
        $utama = ($proc | Sort-Object CreationDate | Select-Object -First 1).ProcessId
        Write-Host "Matiin  :  Ctrl+C di terminalnya, atau  Stop-Process -Id $utama -Force" -ForegroundColor Cyan
    }
}
Write-Host "Log     :  Get-Content '$LogFile' -Tail 20 -Wait`n" -ForegroundColor Cyan
