<#
.SYNOPSIS
  Setup sekali jalan: unduh SEMUA bobot model yang dipakai agent, biar
  OFFLINE_MODE=true beneran bisa jalan tanpa jaringan sama sekali.

  Yang diambil:
    1. Ollama qwen2.5:7b     (~4.7 GB) - otak
    2. Kokoro-82M ONNX       (~337 MB) - suara
    3. Parakeet TDT 0.6B v2  (~660 MB) - pendengaran

  Aman diulang: yang udah ada dilewat.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

.EXAMPLE
  # Sekalian voice Piper (cuma perlu kalau TTS_BACKEND=piper)
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Piper
#>
[CmdletBinding()]
param(
    [string]$OllamaModel = "qwen2.5:7b",
    [switch]$Piper,
    [string]$PiperVoice = "en_US-lessac-medium",
    [string]$PiperPath = "en/en_US/lessac/medium"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ModelsDir = Join-Path $RepoRoot "models"
$Python = Join-Path $RepoRoot ".venv-agent\Scripts\python.exe"

Write-Host "=== personal-agent setup ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot"
if (-not (Test-Path $ModelsDir)) { New-Item -ItemType Directory -Path $ModelsDir | Out-Null }

function Get-Berkas {
    param([string]$Url, [string]$Dest, [string]$Label)
    if ((Test-Path $Dest) -and ((Get-Item $Dest).Length -gt 0)) {
        Write-Host "  $Label sudah ada, skip." -ForegroundColor Green
        return
    }
    Write-Host "  download $Label ..."
    # Invoke-WebRequest jauh lebih cepat tanpa progress bar
    $prev = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    try {
        # Tulis ke .part dulu: kalau unduhannya putus di tengah, yang ketinggalan
        # bukan file utuh-tapi-korup yang bakal dilewat pas dijalanin ulang.
        Invoke-WebRequest -Uri $Url -OutFile "$Dest.part" -UseBasicParsing
        Move-Item "$Dest.part" $Dest -Force
    } finally {
        $ProgressPreference = $prev
        if (Test-Path "$Dest.part") { Remove-Item "$Dest.part" -Force }
    }
    $mb = [math]::Round((Get-Item $Dest).Length / 1MB, 1)
    Write-Host "  selesai ($mb MB)." -ForegroundColor Green
}

# --- 1. Otak: Ollama ---
Write-Host "`n[1/4] Ollama: $OllamaModel" -ForegroundColor Cyan
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "ollama nggak ketemu di PATH. Install dulu dari https://ollama.com/download"
}
$installed = (ollama list) -join "`n"
if ($installed -match [regex]::Escape($OllamaModel)) {
    Write-Host "  sudah ada, skip." -ForegroundColor Green
} else {
    Write-Host "  pull (bisa beberapa menit, ~4.7 GB)..."
    ollama pull $OllamaModel
    if ($LASTEXITCODE -ne 0) { throw "ollama pull gagal (exit $LASTEXITCODE)" }
    Write-Host "  selesai." -ForegroundColor Green
}

# --- 2. Suara: Kokoro ---
Write-Host "`n[2/4] Kokoro-82M (TTS)" -ForegroundColor Cyan
$KokoroBase = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
Get-Berkas "$KokoroBase/kokoro-v1.0.onnx" (Join-Path $ModelsDir "kokoro-v1.0.onnx") "kokoro-v1.0.onnx"
Get-Berkas "$KokoroBase/voices-v1.0.bin"  (Join-Path $ModelsDir "voices-v1.0.bin")  "voices-v1.0.bin"

# --- 3. Pendengaran: Parakeet ---
# Beda dari dua di atas: bobotnya nggak ditaruh di models/, tapi di cache
# HuggingFace, karena onnx-asr yang ngurus sendiri lewat nama model. Cara paling
# jujur buat mastiin lengkap adalah nyuruh onnx-asr muat modelnya sekali.
Write-Host "`n[3/4] Parakeet TDT 0.6B v2 (STT)" -ForegroundColor Cyan
if (-not (Test-Path $Python)) {
    Write-Host "  venv nggak ketemu di $Python — lewati." -ForegroundColor Yellow
} else {
    Write-Host "  muat sekali biar ke-cache (~660 MB kalau belum ada)..."
    & $Python -c "import onnx_asr, agent.config as c; onnx_asr.load_model(c.STT_MODEL); print('  ok, Parakeet siap.')"
    if ($LASTEXITCODE -ne 0) { throw "gagal nyiapin Parakeet (exit $LASTEXITCODE)" }
}

# --- 4. Piper (opsional) ---
Write-Host "`n[4/4] Voice Piper (opsional)" -ForegroundColor Cyan
if (-not $Piper) {
    Write-Host "  dilewat (pakai -Piper kalau TTS_BACKEND=piper)." -ForegroundColor Green
} else {
    $BaseUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/$PiperPath"
    foreach ($f in @("$PiperVoice.onnx", "$PiperVoice.onnx.json")) {
        Get-Berkas "$BaseUrl/$f`?download=true" (Join-Path $ModelsDir $f) $f
    }
}

Write-Host "`nSetup beres." -ForegroundColor Cyan
Write-Host "Cek kesiapan offline:  .\.venv-agent\Scripts\python.exe -m agent.cek_offline"
Write-Host "Jalanin agent:         .\.venv-agent\Scripts\python.exe -m agent.main"
