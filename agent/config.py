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

# --- Otak: "ollama" (lokal, gratis) atau "claude" (API, butuh internet + kunci) ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()

# --- Claude API ---
# Dibaca eksplisit di sini, bukan dibiarin SDK-nya nyomot sendiri dari env:
# kalau implisit, salah nama variabel baru ketahuan pas dipakai — dan yang
# muncul cuma error otentikasi yang nggak nunjuk ke penyebabnya.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
# low/medium/high/xhigh/max. Buat obrolan pendek, "low" paling kenceng.
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "low")
# "disabled" / "adaptive" / kosong (jangan kirim parameternya).
# Default disabled: balasan di sini cuma 1-2 kalimat, mikir dalam nggak nambah
# kualitas tapi nambah jeda — dan di Sonnet 5 thinking NYALA kalau parameternya
# nggak dikirim, yang bikin max_tokens kebagi dua sama isi pikirannya.
CLAUDE_THINKING = os.getenv("CLAUDE_THINKING", "disabled").strip().lower()
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "1024"))
CLAUDE_TIMEOUT = float(os.getenv("CLAUDE_TIMEOUT", "60"))

# --- Ollama ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
# Batas panjang balasan. Bukan buat maksa pendek — prompt-nya udah minta 1-2
# kalimat — tapi jaring pengaman: qwen kadang ngelantur jadi paragraf, dan tiap
# token nyangkut jadi ~0.4 detik audio yang harus didengerin sampai habis.
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "160"))
# Ollama defaultnya 4096 kalau nggak diminta. System prompt aja udah ~850 token
# (jadwal + fakta), jadi 4096 kepakai riwayat beberapa giliran doang, dan yang
# kebuang duluan justru jadwalnya.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# --- Bahasa ---
# Menentukan pengurai tanggal, kata kunci niat, dan prompt yang dipakai.
LANGUAGE = os.getenv("LANGUAGE", "en").lower()

# --- Mode offline ---
# Kalau true, backend yang butuh internet ditolak saat startup — bukan dibiarkan
# gagal diam-diam pas dipakai.
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "true").lower() in ("1", "true", "yes")

# --- STT ---
# "parakeet" = onnx-asr, CPU, nol VRAM, English saja
# "whisper"  = faster-whisper, 99 bahasa, butuh VRAM kalau di GPU
STT_BACKEND = os.getenv("STT_BACKEND", "parakeet").lower()
STT_MODEL = os.getenv("STT_MODEL", "nemo-parakeet-tdt-0.6b-v2")
# cpu / cuda. Parakeet di CPU secepat Whisper di GPU dan nggak makan VRAM sama
# sekali, jadi VRAM-nya bisa dipakai penuh sama LLM.
STT_DEVICE = os.getenv("STT_DEVICE", "cpu").lower()

# --- Whisper (dipakai kalau STT_BACKEND=whisper) ---
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
# Muat model pas agent nyala (bukan nunggu pertanyaan pertama). Defaultnya
# ngikutin setelan di atas: kalau model emang bakal dilepas pas nganggur,
# muat di awal cuma bikin VRAM kepakai percuma sampai ambang idle kelewat.
WHISPER_WARMUP = os.getenv(
    "WHISPER_WARMUP", "false" if WHISPER_IDLE_UNLOAD_SECONDS > 0 else "true"
).lower() in ("1", "true", "yes")
# Contoh gaya bicara + kosakata yang sering kepakai. Whisper mencondongkan
# tebakannya ke kata-kata di sini, jadi nama matkul & istilah teknis nggak
# dikira kata lain. Nol biaya waktu proses. Tambahin nama orang/tempat/tool
# yang sering kamu sebut. Cuma kepakai kalau STT_BACKEND=whisper — Parakeet
# nggak nerima prompt.
WHISPER_PROMPT = os.getenv(
    "WHISPER_PROMPT",
    "Casual conversation with technical terms: Git, commit, push, pull request, "
    "branch, merge, repo, Python, JavaScript, Whisper, Ollama, Kokoro, Parakeet, "
    "VS Code, terminal, API, database, deploy, debug, laptop, campus, ANU, "
    "supervisor, assignment, tutorial, lecture, deadline, meeting, schedule.",
)

# --- TTS ---
# "kokoro" = kokoro-onnx, 54 suara, 24 kHz, CPU
# "piper"  = piper-tts, suara per-bahasa, CPU
TTS_BACKEND = os.getenv("TTS_BACKEND", "kokoro").lower()

# --- Kokoro (dipakai kalau TTS_BACKEND=kokoro) ---
KOKORO_MODEL = _path(os.getenv("KOKORO_MODEL", "models/kokoro-v1.0.onnx"))
KOKORO_VOICES = _path(os.getenv("KOKORO_VOICES", "models/voices-v1.0.bin"))
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))
KOKORO_LANG = os.getenv("KOKORO_LANG", "en-us")

# --- Piper (dipakai kalau TTS_BACKEND=piper) ---
PIPER_VOICE = _path(os.getenv("PIPER_VOICE", "models/id_ID-news_tts-medium.onnx"))

# --- Audio ---
# Batas kata yang ditempel ke system prompt. Beda dari OLLAMA_NUM_PREDICT: itu
# motong paksa di tengah kata, ini bikin modelnya sendiri yang ngerem.
# Terukur di qwen2.5:7b — 41 kata jadi 17, audio 18,4 detik jadi 9,3. Penting
# banget di mode sesi: tanpa barge-in, balasan panjang nggak bisa dipotong.
# 0 = matiin.
REPLY_MAX_WORDS = int(os.getenv("REPLY_MAX_WORDS", "25"))

# --- Mode sesi (ngobrol kontinu) ---
# Sekali pencet hotkey buat masuk sesi, terus ngomong bebas tanpa mencet lagi.
# Batas kalimat dideteksi VAD, bukan tombol.
SESSION_MODE = os.getenv("SESSION_MODE", "false").lower() in ("1", "true", "yes")
# Sesi nutup sendiri kalau sekian detik nggak ada suara. Wajib ada: tanpa ini,
# lupa nutup berarti model nyangkut di memori seharian.
SESSION_IDLE_SECONDS = float(os.getenv("SESSION_IDLE_SECONDS", "30"))
# Peluang minimum sebuah frame dianggap suara (0..1). Naikin kalau kebisingan
# ruangan kebaca sebagai omongan.
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
# Sepi selama ini = kalimatmu dianggap selesai. Kekecilan bikin kalimat
# kepotong di jeda alami; kegedean bikin agent kerasa lelet nyaut.
VAD_SILENCE_MS = float(os.getenv("VAD_SILENCE_MS", "800"))
# Suara lebih pendek dari ini diabaikan — batuk, ketukan keyboard, decak.
VAD_MIN_SPEECH_MS = float(os.getenv("VAD_MIN_SPEECH_MS", "250"))
# Audio sebelum suara kedeteksi yang tetap disimpen. VAD selalu telat sedikit
# ngenalin awal kata; tanpa ini konsonan pertama kepotong.
VAD_SPEECH_PAD_MS = float(os.getenv("VAD_SPEECH_PAD_MS", "300"))

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
# Jumlah pesan (di luar system prompt) yang dibawa ke tiap permintaan.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

# MEMORY_DIR masih kepakai walau fitur memori dicopot: di situ tempat
# agent.lock (kunci satu-instance) dan cache model ditaruh.
MEMORY_DIR = _path(os.getenv("MEMORY_DIR", "memory"))

SYSTEM_PROMPT = """You are a personal assistant. Speak casually, like a friend.

You are talking BY VOICE, not text. What you receive is a transcript of the
user speaking into a microphone, and your reply is read back through a speaker.
So you really do "hear" and "speak" — never say you cannot hear, and never ask
the user to type. If a transcript looks misheard or cut off, guess the most
sensible meaning from context, or ask them to repeat it.

Important rules:
- Your reply is READ ALOUD, so keep it short: one or two sentences at most.
- No bullet points, numbering, markdown, emoji, or formatting of any kind.
  Plain sentences only.
- No abbreviations or symbols that are awkward to say out loud.
- If you don't know, say so. Don't make things up.
- If something needs a long answer, give the gist first and offer to go on.

NEVER claim to have done something you did not do. You have NO tools: you cannot
read or change a calendar, save notes or tasks, set reminders, send anything, or
look anything up. You can only talk. If asked to do any of that, say plainly that
you can't do it — don't say "done", "I've added it", or "I'll remind you".

You also have no clock and no calendar, so you don't know today's date or the
time unless the user tells you in this conversation. Say so rather than guessing.

A wrong answer still gets caught when the user checks. A false claim of having
acted makes them stop checking, which is worse.
"""


# --- Penegakan mode offline ---------------------------------------------------
# Ditaruh di config, bukan di main.py, biar skrip dan tes ikut kena — setelan
# yang cuma diperiksa di satu entry point itu setelan yang gampang bocor.


class OfflineViolation(RuntimeError):
    """Setelan minta jaringan padahal OFFLINE_MODE=true."""


def _cek_offline() -> list[str]:
    """Daftar setelan yang bentrok sama mode offline. Kosong = aman."""
    if not OFFLINE_MODE:
        return []

    masalah = []
    if LLM_BACKEND == "claude":
        masalah.append(
            "LLM_BACKEND=claude butuh Anthropic API. Pakai LLM_BACKEND=ollama, "
            "atau matikan OFFLINE_MODE."
        )
    if STT_BACKEND not in ("parakeet", "whisper"):
        masalah.append(f"STT_BACKEND={STT_BACKEND!r} bukan backend lokal.")
    if TTS_BACKEND not in ("kokoro", "piper"):
        masalah.append(f"TTS_BACKEND={TTS_BACKEND!r} bukan backend lokal.")
    return masalah


def wajib_offline() -> None:
    """Berhenti sekarang kalau setelannya butuh jaringan.

    Sengaja gagal di startup, bukan pas dipakai: agent jalan tanpa jendela,
    jadi kegagalan di tengah percakapan cuma kedengeran kayak agent bisu.
    """
    masalah = _cek_offline()
    if masalah:
        raise OfflineViolation(
            "OFFLINE_MODE=true tapi setelannya butuh jaringan:\n"
            + "\n".join(f"  - {m}" for m in masalah)
        )
