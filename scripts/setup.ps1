<#
.SYNOPSIS
  One-time setup: download EVERY model weight the agent uses, so that
  OFFLINE_MODE=true can genuinely run with no network at all.

  What it fetches:
    1. Ollama qwen2.5:7b     (~4.7 GB) - the brain
    2. Kokoro-82M ONNX       (~337 MB) - the voice
    3. Parakeet TDT 0.6B v2  (~660 MB) - the ears
    4. Silero VAD            (~2 MB)   - utterance boundaries

  Safe to re-run: anything already present is skipped.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

.EXAMPLE
  # Also fetch a Piper voice (only needed when TTS_BACKEND=piper)
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

function Get-File {
    param([string]$Url, [string]$Dest, [string]$Label)
    if ((Test-Path $Dest) -and ((Get-Item $Dest).Length -gt 0)) {
        Write-Host "  $Label already present, skipping." -ForegroundColor Green
        return
    }
    Write-Host "  downloading $Label ..."
    # Invoke-WebRequest is far faster without the progress bar
    $prev = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    try {
        # Write to .part first: if the download is interrupted, what is left
        # behind is not a whole-looking but corrupt file that a re-run skips.
        Invoke-WebRequest -Uri $Url -OutFile "$Dest.part" -UseBasicParsing
        Move-Item "$Dest.part" $Dest -Force
    } finally {
        $ProgressPreference = $prev
        if (Test-Path "$Dest.part") { Remove-Item "$Dest.part" -Force }
    }
    $mb = [math]::Round((Get-Item $Dest).Length / 1MB, 1)
    Write-Host "  done ($mb MB)." -ForegroundColor Green
}

# --- 1. Brain: Ollama ---
Write-Host "`n[1/5] Ollama: $OllamaModel" -ForegroundColor Cyan
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "ollama not found on PATH. Install it from https://ollama.com/download"
}
$installed = (ollama list) -join "`n"
if ($installed -match [regex]::Escape($OllamaModel)) {
    Write-Host "  already present, skipping." -ForegroundColor Green
} else {
    Write-Host "  pulling (several minutes, ~4.7 GB)..."
    ollama pull $OllamaModel
    if ($LASTEXITCODE -ne 0) { throw "ollama pull failed (exit $LASTEXITCODE)" }
    Write-Host "  done." -ForegroundColor Green
}

# --- 2. Voice: Kokoro ---
Write-Host "`n[2/5] Kokoro-82M (TTS)" -ForegroundColor Cyan
$KokoroBase = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
Get-File "$KokoroBase/kokoro-v1.0.onnx" (Join-Path $ModelsDir "kokoro-v1.0.onnx") "kokoro-v1.0.onnx"
Get-File "$KokoroBase/voices-v1.0.bin"  (Join-Path $ModelsDir "voices-v1.0.bin")  "voices-v1.0.bin"

# --- 3. Ears: Parakeet ---
# Unlike the two above, these weights do not live in models/ but in the
# HuggingFace cache, because onnx-asr fetches them by model name. The honest way
# to confirm they are complete is to have onnx-asr load the model once.
Write-Host "`n[3/5] Parakeet TDT 0.6B v2 (STT)" -ForegroundColor Cyan
if (-not (Test-Path $Python)) {
    Write-Host "  venv not found at $Python — skipping." -ForegroundColor Yellow
} else {
    Write-Host "  loading once so it gets cached (~660 MB if not present)..."
    & $Python -c "import onnx_asr, src.config as c; onnx_asr.load_model(c.STT_MODEL); print('  ok, Parakeet ready.')"
    if ($LASTEXITCODE -ne 0) { throw "failed to prepare Parakeet (exit $LASTEXITCODE)" }
}

# --- 4. Utterance boundaries: Silero VAD ---
Write-Host "`n[4/5] Silero VAD" -ForegroundColor Cyan
if (-not (Test-Path $Python)) {
    Write-Host "  venv not found — skipping." -ForegroundColor Yellow
} else {
    & $Python -c "import onnx_asr; onnx_asr.load_vad('silero'); print('  ok, VAD ready.')"
    if ($LASTEXITCODE -ne 0) { throw "failed to prepare the VAD (exit $LASTEXITCODE)" }
}

# --- 5. Piper (optional) ---
Write-Host "`n[5/5] Piper voice (optional)" -ForegroundColor Cyan
if (-not $Piper) {
    Write-Host "  skipped (pass -Piper when TTS_BACKEND=piper)." -ForegroundColor Green
} else {
    $BaseUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/$PiperPath"
    foreach ($f in @("$PiperVoice.onnx", "$PiperVoice.onnx.json")) {
        Get-File "$BaseUrl/$f`?download=true" (Join-Path $ModelsDir $f) $f
    }
}

Write-Host "`nSetup complete." -ForegroundColor Cyan
Write-Host "Check offline readiness:  .\.venv-agent\Scripts\python.exe -m src.offline_check"
Write-Host "Run the agent:            .\.venv-agent\Scripts\python.exe -m src.agent"
