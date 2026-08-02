"""Entry point: register hotkey push-to-talk & wiring pipeline STT -> LLM -> TTS."""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
import threading
import time
from typing import Callable

from . import audio, config, llm, stt, tts

log = logging.getLogger("agent")


# --- Logging ----------------------------------------------------------------


def setup_logging() -> None:
    """Log ke file (wajib: jalan di background tanpa console) + console kalau ada."""
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.handlers.RotatingFileHandler(
        config.LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # pythonw.exe nggak punya stdout — cuma pasang console handler kalau beneran ada
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt)
        root.addHandler(console)

    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Exception yang lolos dari thread mana pun tetep ke-log
    sys.excepthook = lambda *exc: log.critical("uncaught exception", exc_info=exc)
    threading.excepthook = lambda args: log.critical(
        "uncaught exception di thread %s",
        args.thread.name if args.thread else "?",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


# --- Hotkey backends --------------------------------------------------------


class _BackendBase:
    """Bagian yang sama buat semua backend hotkey.

    Dua mode:
    - toggle: pencetan pertama mulai rekam, pencetan kedua berhenti
    - hold:   rekam selama tombol ditahan (push-to-talk)
    """

    name = "?"
    # Pencetan yang lebih rapat dari ini dianggap satu pencetan. Melindungi
    # dari event UP/DOWN palsu yang dikirim Windows selama tombol ditahan.
    EDGE_DEBOUNCE = 0.3

    def __init__(self, combo: str, on_activate: Callable[[Callable[[], bool]], None]):
        self.combo_label = combo
        self.on_activate = on_activate
        self._busy = False  # pipeline lagi jalan
        self._recording = False  # mode toggle: lagi ngerekam
        self._last_edge = 0.0
        self._is_held: Callable[[], bool] = lambda: False

    def _on_edge(self) -> None:
        """Kombinasi hotkey baru aja jadi lengkap (dipanggil sekali per pencetan)."""
        now = time.monotonic()
        if now - self._last_edge < self.EDGE_DEBOUNCE:
            return
        self._last_edge = now

        if config.HOTKEY_MODE == "hold":
            self._launch(self._is_held)
            return

        if self._recording:
            self._recording = False  # pencetan kedua: berhenti rekam
            log.debug("toggle: stop")
        elif not self._busy:
            self._recording = True  # pencetan pertama: mulai rekam
            log.debug("toggle: start")
            self._launch(lambda: self._recording)
        else:
            log.info("masih ngerjain yang sebelumnya, hotkey diabaikan")
            # Jangan diam: user perlu tau dia kedengeran, cuma lagi sibuk
            threading.Thread(target=audio.beep_busy, daemon=True).start()

    def _launch(self, is_recording: Callable[[], bool]) -> None:
        """Jalanin pipeline di thread terpisah.

        WAJIB terpisah: callback backend jalan di thread yang sama dengan yang
        ngeproses event keyboard. Kalau di-block, pencetan kedua (toggle) atau
        event lepas (hold) nggak akan pernah kebaca.
        """
        if self._busy:
            return
        self._busy = True

        def _run():
            try:
                self.on_activate(is_recording)
            finally:
                self._busy = False
                self._recording = False

        threading.Thread(target=_run, name="utterance", daemon=True).start()


class KeyboardBackend(_BackendBase):
    """Backend `keyboard`. Paling akurat, tapi di Windows butuh admin."""

    name = "keyboard"

    def run(self) -> None:
        import keyboard as kb

        self._is_held = lambda: kb.is_pressed(self.combo_label)

        kb.add_hotkey(
            self.combo_label,
            self._on_edge,
            suppress=config.HOTKEY_SUPPRESS,
        )
        log.info(
            "Hotkey '%s' aktif, mode %s (backend keyboard). Ctrl+C buat berhenti.",
            self.combo_label,
            config.HOTKEY_MODE,
        )
        kb.wait()


class PynputBackend(_BackendBase):
    """Backend `pynput`. Jalan tanpa admin, dipilih lewat HOTKEY_BACKEND=pynput.

    Catatan: pynput nggak bisa nge-suppress hotkey, jadi `Ctrl+Space` tetep
    diterusin ke aplikasi yang lagi fokus.
    """

    name = "pynput"

    _ALIASES = {
        "control": "ctrl",
        "win": "cmd",
        "super": "cmd",
        "meta": "cmd",
        "esc": "escape",
        "return": "enter",
        "rctrl": "right ctrl",
        "lctrl": "left ctrl",
        "ctrl_r": "right ctrl",
        "ctrl_l": "left ctrl",
        "ralt": "right alt",
        "lalt": "left alt",
        "rshift": "right shift",
        "lshift": "left shift",
    }

    # Suffix pynput -> kata sisi
    _SIDES = {"_l": "left", "_r": "right", "_gr": "right"}

    def __init__(self, combo: str, on_activate: Callable[[Callable[[], bool]], None]):
        super().__init__(combo, on_activate)
        self.combo = {
            self._ALIASES.get(p.strip().lower(), p.strip().lower())
            for p in combo.split("+")
            if p.strip()
        }
        # Simpan key mentah yang lagi ditahan; nama-namanya dihitung pas dibutuhin,
        # biar nahan ctrl kiri & kanan bareng nggak bikin state-nya kacau.
        self._pressed: set[str] = set()
        self._is_held = self._combo_complete
        self._combo_was_complete = False

    def _raw_name(self, key) -> str | None:
        from pynput import keyboard as pk

        if isinstance(key, pk.Key):
            return key.name
        if isinstance(key, pk.KeyCode):
            if key.char:
                return key.char.lower()
            # Pas modifier ditahan, char-nya jadi control character. Pakai vk.
            if key.vk is not None and 65 <= key.vk <= 90:
                return chr(key.vk).lower()
        return None

    def _names(self, raw: str) -> set[str]:
        """Nama-nama yang dipenuhi satu tombol.

        ctrl_r cocok buat 'ctrl' maupun 'right ctrl', jadi user bisa milih mau
        spesifik sisi atau nggak.
        """
        for suffix, side in self._SIDES.items():
            if raw.endswith(suffix):
                base = raw[: -len(suffix)]
                base = self._ALIASES.get(base, base)
                return {base, f"{side} {base}"}
        return {self._ALIASES.get(raw, raw)}

    def _combo_complete(self) -> bool:
        active: set[str] = set()
        for raw in tuple(self._pressed):
            active |= self._names(raw)
        return self.combo <= active

    def _on_press(self, key) -> None:
        raw = self._raw_name(key)
        if raw is None:
            return
        self._pressed.add(raw)
        # Cuma picu pas kombinasi baru jadi lengkap, bukan tiap auto-repeat
        if self._combo_complete() and not self._combo_was_complete:
            self._combo_was_complete = True
            self._on_edge()

    def _on_release(self, key) -> None:
        raw = self._raw_name(key)
        if raw is not None:
            self._pressed.discard(raw)
        if not self._combo_complete():
            self._combo_was_complete = False

    def run(self) -> None:
        from pynput import keyboard as pk

        log.info(
            "Hotkey '%s' aktif, mode %s (backend pynput). Ctrl+C buat berhenti.",
            self.combo_label,
            config.HOTKEY_MODE,
        )
        with pk.Listener(on_press=self._on_press, on_release=self._on_release) as ln:
            ln.join()


def make_backend(on_activate):
    if config.HOTKEY_BACKEND == "keyboard":
        return KeyboardBackend(config.HOTKEY, on_activate)
    if config.HOTKEY_BACKEND != "pynput":
        log.warning(
            "HOTKEY_BACKEND '%s' nggak dikenal, balik ke 'pynput'",
            config.HOTKEY_BACKEND,
        )
    return PynputBackend(config.HOTKEY, on_activate)


# --- Pipeline ---------------------------------------------------------------

# Frasa penghapus memori. Dicocokkan lokal, bukan lewat LLM: perintah yang
# nggak bisa dibatalkan nggak boleh gantung pada tebakan model.
_FRASA_LUPA = (
    "lupakan semua",
    "lupain semua",
    "hapus memori",
    "hapus ingatan",
    "hapus semua memori",
    "lupakan semuanya",
)


def _minta_dilupakan(text: str) -> bool:
    bersih = re.sub(r"[^\w\s]", " ", text.lower())
    bersih = re.sub(r"\s+", " ", bersih).strip()
    return any(f in bersih for f in _FRASA_LUPA)


def handle_utterance(is_recording: Callable[[], bool]) -> None:
    """Satu putaran: rekam -> transcribe -> LLM -> ngomong."""
    # 0. Kalau model lagi terlepas, mulai muat SEKARANG — barengan sama user
    #    ngomong. Durasi ngomong (biasanya 3-5 detik) jadi kepakai buat muat,
    #    bukan nganggur. get_model() punya lock sendiri, jadi transcribe di
    #    bawah bakal nunggu dengan rapi kalau muatnya belum kelar.
    if not stt.is_loaded():
        threading.Thread(target=stt.warmup, name="muat-duluan", daemon=True).start()

    # 1. Rekam
    try:
        # beep dibunyiin dari dalam, pas mic udah beneran siap
        clip = audio.record_until_release(
            is_recording,
            on_ready=audio.beep_start,
            # Mode toggle berhenti eksplisit, nggak perlu tenggang anti-kedip
            release_grace=0.0 if config.HOTKEY_MODE == "toggle" else None,
        )
        audio.beep_stop()
    except Exception:
        log.exception("gagal merekam mic")
        audio.beep_error()
        return

    if clip.size == 0:
        log.info("nggak ada audio yang kepakai, skip")
        return

    # 2. STT
    # Muat model bisa makan 10-35 detik kalau file-nya dingin. Tanpa kabar apa
    # pun, diam selama itu nggak bisa dibedain dari mati — user bakal mencet
    # ulang dan makin bingung. Piper udah di-warmup jadi ngomongnya instan.
    if not stt.is_loaded():
        log.info("model masih dimuat, ngabarin user dulu")
        _say_safely("Sebentar ya, lagi nyiapin.")

    try:
        text = stt.transcribe(clip)
    except Exception:
        log.exception("gagal transcribe")
        audio.beep_error()
        return

    if not text:
        log.info("transcript kosong, skip")
        audio.beep_error()
        return
    log.info("User: %s", text)

    # 2b. Perintah hapus memori — ditangani lokal, nggak dikirim ke LLM
    if _minta_dilupakan(text):
        llm.get_conversation().forget()
        _say_safely("Oke, semua yang aku inget tentang kamu udah aku hapus.")
        return

    # 3. LLM
    try:
        reply = llm.chat(text)
    except Exception:
        log.exception("gagal manggil LLM (%s)", config.LLM_BACKEND)
        _say_safely("Maaf, otakku lagi nggak nyambung.")
        return

    # 4. TTS + playback
    try:
        audio.play_wav(tts.speak(reply))
    except Exception:
        log.exception("gagal ngomong")
        audio.beep_error()


def _say_safely(text: str) -> None:
    """Ngomong tanpa bikin error baru (dipakai buat ngabarin kegagalan)."""
    try:
        audio.play_wav(tts.speak(text))
    except Exception:
        log.exception("gagal ngomong pesan error")
        audio.beep_error()


def warmup() -> None:
    """Load model di background biar pencetan pertama nggak lama.

    Piper selalu dimuat: CPU-only, 0 VRAM, cuma ~1.5 detik. Whisper tergantung
    WHISPER_WARMUP — kalau modelnya toh bakal dilepas pas nganggur, muat di awal
    cuma nahan ~1.9 GB VRAM percuma sampai ambang idle kelewat.
    """

    def _run():
        try:
            tts.get_voice()
            if config.WHISPER_WARMUP:
                stt.warmup()
            else:
                log.info(
                    "Whisper nggak di-warmup; dimuat pas pertanyaan pertama "
                    "(+~4 detik sekali)"
                )
            log.info("Warmup selesai, siap dipakai.")
        except Exception:
            log.exception("warmup gagal (model bakal di-load pas dipakai)")

    threading.Thread(target=_run, name="warmup", daemon=True).start()


def main() -> int:
    setup_logging()
    log.info("=" * 60)
    otak = (
        config.OLLAMA_MODEL if config.LLM_BACKEND == "ollama" else config.CLAUDE_MODEL
    )
    log.info(
        "personal-agent start | hotkey=%s (%s, %s) | whisper=%s/%s | llm=%s/%s",
        config.HOTKEY,
        config.HOTKEY_MODE,
        config.HOTKEY_BACKEND,
        config.WHISPER_MODEL,
        config.WHISPER_DEVICE,
        config.LLM_BACKEND,
        otak,
    )

    warmup()
    backend = make_backend(handle_utterance)

    try:
        backend.run()
    except KeyboardInterrupt:
        log.info("dihentikan lewat Ctrl+C")
    except ImportError:
        log.exception("backend hotkey '%s' nggak bisa di-import", backend.name)
        return 1
    except Exception:
        log.exception("hotkey listener mati")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
