"""Semua konstanta & system prompt. Bisa di-override lewat file .env di root repo."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")


def _path(value: str) -> Path:
    """Path relatif dianggap relatif ke root repo, biar aman dipanggil dari mana aja."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


# --- Hotkey ---
HOTKEY = os.getenv("HOTKEY", "ctrl+space")
# "toggle" = pencet sekali mulai, pencet lagi berhenti.
# "hold"   = tahan selama ngomong (push-to-talk).
HOTKEY_MODE = os.getenv("HOTKEY_MODE", "toggle").lower()
# "pynput" = jalan tanpa admin (default, terverifikasi di mesin ini).
# "keyboard" = bisa nge-suppress hotkey, tapi butuh admin dan di sebagian mesin
# dispatch-nya nggak jalan (add_hotkey nggak pernah kepanggil).
HOTKEY_BACKEND = os.getenv("HOTKEY_BACKEND", "pynput").lower()
# Telen hotkey-nya biar nggak nyasar ke aplikasi yang lagi fokus.
# Cuma didukung backend "keyboard"; pynput selalu nerusin.
HOTKEY_SUPPRESS = os.getenv("HOTKEY_SUPPRESS", "true").lower() in ("1", "true", "yes")

# --- Otak: "claude" (API, butuh internet + kunci) atau "ollama" (lokal, gratis) ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "claude").lower()

# --- Claude API ---
# Kunci diambil otomatis dari env ANTHROPIC_API_KEY (bisa ditaruh di .env)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
# low/medium/high/xhigh/max. Buat obrolan pendek, "low" paling kenceng.
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "low")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "1024"))
CLAUDE_TIMEOUT = float(os.getenv("CLAUDE_TIMEOUT", "60"))

# --- Ollama ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# --- Whisper (STT) ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
WHISPER_LANG = os.getenv("WHISPER_LANG", "id")
# Lepas model dari VRAM/RAM kalau nganggur selama ini (detik). 0 = nggak pernah.
# Model medium megang ~2 GB VRAM terus-terusan padahal cuma kepakai 0.5 detik per
# kalimat — di GPU pribadi yang dipakai buat hal lain, itu sayang. Ollama juga
# begini defaultnya (keep-alive 5 menit). Bayarannya: muat ulang ~7 detik di
# pertanyaan pertama setelah nganggur lama.
WHISPER_IDLE_UNLOAD_SECONDS = float(os.getenv("WHISPER_IDLE_UNLOAD_SECONDS", "900"))
# Contoh gaya bicara + kosakata yang sering kepakai. Whisper mencondongkan tebakannya
# ke kata-kata di sini, jadi istilah teknis Inggris nggak dikira kata Indonesia.
# Nol biaya waktu proses. Tambahin nama orang/tempat/tool yang sering kamu sebut.
WHISPER_PROMPT = os.getenv(
    "WHISPER_PROMPT",
    "Ngobrol santai campur istilah teknis: Git, commit, push, pull request, branch, "
    "merge, repo, Python, JavaScript, Whisper, Ollama, Piper, VS Code, terminal, "
    "API, database, deploy, debug, error, laptop, kampus, ANU, dosen pembimbing, "
    "tugas kuliah, deadline, meeting, jadwal.",
)

# --- Piper (TTS) ---
PIPER_VOICE = _path(os.getenv("PIPER_VOICE", "models/id_ID-news_tts-medium.onnx"))

# --- Audio ---
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = 1
# Rekaman lebih pendek dari ini dianggap salah pencet dan dibuang
MIN_RECORD_SECONDS = float(os.getenv("MIN_RECORD_SECONDS", "0.4"))
# Batas aman biar hotkey nyangkut nggak bikin RAM habis
MAX_RECORD_SECONDS = float(os.getenv("MAX_RECORD_SECONDS", "60"))
# Tenggang sebelum rekaman beneran distop pas hotkey kedeteksi lepas.
# Windows suka ngirim pasangan UP/DOWN palsu selama tombol ditahan (auto-repeat),
# jadi tanpa tenggang ini rekamannya kepotong-potong jadi serpihan.
RELEASE_GRACE_SECONDS = float(os.getenv("RELEASE_GRACE_SECONDS", "0.6"))
# Bantalan hening di awal & akhir tiap playback. Windows suka motong ujung suara:
# device output butuh waktu buka (awal keburu jalan) dan buffer internalnya masih
# main pas sounddevice udah nganggap selesai (akhir kepotong).
# Diukur di mesin ini: tanpa bantalan, 119 ms hilang dari nada 2 detik.
PLAYBACK_PAD_SECONDS = float(os.getenv("PLAYBACK_PAD_SECONDS", "0.2"))

# --- Log ---
LOG_FILE = _path(os.getenv("LOG_FILE", "logs/agent.log"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Simpan tiap rekaman mic ke logs/rec/*.wav. Buat nyetel STT tanpa perlu ngulang
# ngomong terus-terusan. Matiin kalau udah nggak perlu — ini nyimpen suaramu.
SAVE_RECORDINGS = os.getenv("SAVE_RECORDINGS", "false").lower() in ("1", "true", "yes")
RECORDINGS_DIR = _path(os.getenv("RECORDINGS_DIR", "logs/rec"))

# --- Percakapan ---
# Jumlah pesan (di luar system prompt) yang disimpan di memori. History ephemeral,
# hilang pas proses mati. TODO(tahap 2): persist ke disk biar nyambung antar sesi.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

SYSTEM_PROMPT = """Kamu asisten pribadi yang ngobrol dalam Bahasa Indonesia santai, kayak temen.

Kamu lagi ngobrol LEWAT SUARA, bukan teks. Yang kamu terima itu hasil transkrip
omongan user dari mikrofon, dan jawabanmu dibacakan balik lewat speaker. Jadi kamu
memang "dengar" dan "ngomong" — jangan pernah bilang kamu nggak bisa mendengar atau
minta user ngetik. Kalau transkripnya kelihatan salah dengar atau kepotong, tebak
maksudnya dari konteks, atau minta user ngulang.

Aturan penting:
- Jawabanmu DIBACAKAN lewat speaker, jadi WAJIB singkat: maksimal 1-2 kalimat.
- Jangan pakai bullet point, nomor, markdown, emoji, atau format apa pun. Teks polos aja.
- Jangan pakai singkatan aneh atau simbol yang susah dibaca suara.
- Kalau nggak tahu, bilang nggak tahu. Jangan ngarang.
- Kalau butuh jawaban panjang, kasih intinya dulu terus tawarin lanjut.
"""
