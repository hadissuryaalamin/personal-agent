# personal-agent — Build Plan (Tahap 1)

Spec ini buat dikerjain Claude Code. Bangun sesuai urutan di bagian **Build order**.
Kalau ada keputusan ambigu, ikutin default yang udah ditulis di sini; jangan berhenti nanya
kecuali beneran mentok.

## Tujuan
Voice assistant lokal yang jalan **di background** di Windows. Interaksi **push-to-talk**
lewat hotkey: tahan tombol → ngomong → lepas → agent jawab pakai suara. Ngobrol dalam
**Bahasa Indonesia**. Otaknya LLM lokal via Ollama. Nggak ada UI window.

## Scope
**Masuk tahap 1:** hotkey push-to-talk, STT lokal, chat via Ollama, TTS lokal, jalan
diam di background saat login, logging ke file.

**BUKAN tahap 1 (jangan dibangun dulu, cukup sisakan hook/TODO):** baca daftar tugas
otomatis, wake word ("hey ..."), memori antar-sesi (persist history ke disk), notifikasi
proaktif, integrasi kalender/LMS.

## Arsitektur (two-tier)
Lapisan ringan (hotkey listener) jalan terus. Pipeline berat (STT → LLM → TTS) cuma
nyala pas dipicu.

```
tahan hotkey → rekam mic → faster-whisper (id) → Ollama (qwen2.5:7b) → piper (id_ID) → speaker
```

## Stack
- Python 3.11+ (kelola via mise; sediakan juga venv fallback)
- `sounddevice` + `numpy` + `soundfile` — capture mic & playback
- `keyboard` — global hotkey push-to-talk (deteksi press & release).
  Catatan: di Windows sering butuh admin. Alternatif tanpa admin: `pynput`.
  Default pakai `keyboard`; kalau butuh no-admin, sediakan flag buat swap ke `pynput`.
- `faster-whisper` — STT
- Ollama HTTP API (`http://localhost:11434/api/chat`), model `qwen2.5:7b` — udah terinstall
- `piper-tts` + voice `id_ID-news_tts-medium` (dari `rhasspy/piper-voices` di HuggingFace)

## Interaksi
Push-to-talk: **tahan `Ctrl+Space`** → rekam selama ditahan → **lepas** → transcribe →
kirim ke LLM → jawab lewat speaker. Kasih *beep* pendek pas mulai & selesai rekam sebagai
feedback (karena nggak ada window). Hotkey harus bisa diganti lewat config.

## Struktur repo
```
personal-agent/
  agent/
    __init__.py
    config.py        # semua konstanta & system prompt
    audio.py         # record_until_release(), play_wav()
    stt.py           # wrapper faster-whisper
    llm.py           # client Ollama + riwayat percakapan (in-memory)
    tts.py           # wrapper piper
    main.py          # register hotkey & wiring pipeline
  models/            # file voice piper (gitignore)
  scripts/
    setup.ps1            # pull model ollama + download voice piper
    install-startup.ps1  # daftarin Task Scheduler (pythonw, at log on)
  pyproject.toml
  .env.example
  .gitignore
  README.md
```

## Detail per komponen
- **config.py** — `HOTKEY="ctrl+space"`, `OLLAMA_MODEL="qwen2.5:7b"`,
  `OLLAMA_URL`, `WHISPER_MODEL="small"`, `WHISPER_DEVICE="cpu"`,
  `WHISPER_COMPUTE="int8"`, `WHISPER_LANG="id"`, `PIPER_VOICE` (path .onnx),
  `SAMPLE_RATE=16000`, dan `SYSTEM_PROMPT` (Bahasa Indonesia, santai, **wajib jawab
  singkat 1–2 kalimat** karena dibacakan lewat speaker).
- **audio.py** — `record_until_release(is_held) -> np.ndarray` pakai `sounddevice`
  InputStream, buffer frame selama hotkey ditahan. `play_wav(wav_bytes)`. Plus
  `beep_start()` / `beep_stop()`.
- **stt.py** — load model faster-whisper sekali (lazy singleton).
  `transcribe(audio) -> str` dengan `language="id"`.
- **llm.py** — simpan `messages` (mulai dari system prompt). `chat(text) -> str`:
  append user msg, POST ke `/api/chat` dengan `stream=false`, append balasan, return teks.
  History cukup in-memory (ephemeral) untuk tahap 1.
- **tts.py** — `speak(text) -> wav_bytes` jalanin piper (subprocess CLI atau lib python)
  dengan voice `id_ID`. CPU-only.
- **main.py** — register hotkey; on press → beep + mulai rekam; on release → beep +
  stt → llm → tts → play. Bungkus tiap tahap dengan try/except, **log semua error ke file**
  (background nggak punya console). Loop terus sampai proses dimatiin.

## Catatan resource (penting, mesin 8GB VRAM)
- **Whisper**: default `device="cpu", compute_type="int8", model="small"` biar nggak
  rebutan VRAM sama Ollama. Sediakan opsi `cuda` di config kalau user mau lebih cepat.
- **Piper**: CPU, 0 VRAM — aman.
- **Ollama**: biarin keep-alive default (model diturunin dari VRAM saat idle); terima
  warmup beberapa detik di pertanyaan pertama. Sediakan catatan di README soal
  `OLLAMA_KEEP_ALIVE` kalau user mau respons instan.

## Background / startup (Windows)
- Jalanin lewat **`pythonw.exe`** (Python tanpa console window) → prosesnya diam.
- **scripts/install-startup.ps1**: bikin task di Task Scheduler, trigger **At log on**,
  action `pythonw.exe <repo>\agent\main.py`, "Start in" = folder repo, jalanin
  **with highest privileges** (biar `keyboard` bisa nangkep global hotkey).
- **scripts/setup.ps1**: `ollama pull qwen2.5:7b`, download voice piper `id_ID-news_tts-medium`
  (.onnx + .onnx.json) ke `models/`.

## Acceptance criteria (definition of done)
1. Setup script jalan: model Ollama ke-pull, voice Piper ke-download.
2. `python agent/main.py`: tahan `Ctrl+Space`, ngomong Indonesia → transcribe benar.
3. Balasan dari `qwen2.5:7b` kedengeran lewat voice Piper Indonesia.
4. `install-startup.ps1` bikin agent jalan **hidden** saat login; error/aktivitas ke-log ke file.
5. README jelasin: cara install, hotkey, cara ganti model/voice, cara uninstall startup task.

## Build order (buat Claude Code)
1. Scaffold repo: `pyproject.toml`, `config.py`, `.gitignore`, `.env.example`.
2. **tts.py** duluan — paling gampang diverifikasi (teks → speaker).
3. **stt.py** — mic → teks.
4. **llm.py** — Ollama chat + history.
5. **audio.py** + wiring hotkey di **main.py**.
6. Logging ke file + error handling.
7. `setup.ps1`, `install-startup.ps1`, `README.md`.

## Asumsi (koreksi kalau salah)
- Hotkey default `Ctrl+Space` (bisa diganti di config).
- Whisper `small` di CPU int8 sebagai default (seimbang di i5).
- Voice Indonesia Piper cuma satu: `id_ID-news_tts-medium` — itu yang dipakai.
