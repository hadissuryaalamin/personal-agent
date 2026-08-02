<#
.SYNOPSIS
  Setup sekali jalan: pull model Ollama + download voice Piper Bahasa Indonesia.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#>
[CmdletBinding()]
param(
    [string]$OllamaModel = "qwen2.5:7b",
    [string]$Voice = "id_ID-news_tts-medium",
    # Path di dalam repo rhasspy/piper-voices di HuggingFace
    [string]$VoicePath = "id/id_ID/news_tts/medium"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ModelsDir = Join-Path $RepoRoot "models"

Write-Host "=== personal-agent setup ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot"

# --- 1. Model Ollama ---
Write-Host "`n[1/2] Model Ollama: $OllamaModel" -ForegroundColor Cyan
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "ollama nggak ketemu di PATH. Install dulu dari https://ollama.com/download"
}

$installed = (ollama list) -join "`n"
if ($installed -match [regex]::Escape($OllamaModel)) {
    Write-Host "  sudah ada, skip." -ForegroundColor Green
} else {
    Write-Host "  pull (bisa beberapa menit, ~4.7GB)..."
    ollama pull $OllamaModel
    if ($LASTEXITCODE -ne 0) { throw "ollama pull gagal (exit $LASTEXITCODE)" }
    Write-Host "  selesai." -ForegroundColor Green
}

# --- 2. Voice Piper ---
Write-Host "`n[2/2] Voice Piper: $Voice" -ForegroundColor Cyan
if (-not (Test-Path $ModelsDir)) { New-Item -ItemType Directory -Path $ModelsDir | Out-Null }

$BaseUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/$VoicePath"
$files = @("$Voice.onnx", "$Voice.onnx.json")

foreach ($f in $files) {
    $dest = Join-Path $ModelsDir $f
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) {
        Write-Host "  $f sudah ada, skip." -ForegroundColor Green
        continue
    }
    $url = "$BaseUrl/$f`?download=true"
    Write-Host "  download $f ..."
    $prev = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"   # bikin Invoke-WebRequest jauh lebih cepat
    try {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    } finally {
        $ProgressPreference = $prev
    }
    $sizeMb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
    Write-Host "  selesai ($sizeMb MB)." -ForegroundColor Green
}

Write-Host "`nSetup beres." -ForegroundColor Cyan
Write-Host "Coba jalanin:  .\.venv-agent\Scripts\python.exe -m agent.main"
