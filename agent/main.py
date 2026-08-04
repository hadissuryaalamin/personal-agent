"""Entry point: register hotkey push-to-talk & wiring pipeline STT -> LLM -> TTS."""

from __future__ import annotations

import logging
import logging.handlers
import os
import pathlib
import re
import sys
import threading
import time
from typing import Callable

from datetime import datetime

from . import (
    audio,
    calendar,
    config,
    gcal,
    jadwal_baru,
    jawab_pasti,
    kalender_lokal,
    llm,
    stt,
    tts,
    teks,
    tugas,
    vad,
)

log = logging.getLogger("agent")


def _versi_build() -> str:
    """Commit yang lagi jalan + apakah ada file yang lebih baru dari prosesnya.

    Ditulis ke log startup karena udah dua kali kejadian: kode diperbaiki, tapi
    yang diuji proses lama yang masih megang kode lama — Python muat kode pas
    start, ngedit file nggak nyentuh proses yang udah jalan. Gejalanya nyasar,
    kelihatan kayak perbaikannya nggak jalan.
    """
    import subprocess

    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%h %ad", "--date=format:%d/%m %H:%M"],
            cwd=pathlib.Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        commit = r.stdout.strip() or "?"
    except Exception:
        commit = "?"

    # File .py yang lebih baru dari commit-nya = ada perubahan yang belum
    # di-commit; nggak fatal, tapi enak ditandain pas ngebandingin log.
    try:
        terbaru = max(
            p.stat().st_mtime for p in pathlib.Path(__file__).parent.glob("*.py")
        )
        from datetime import datetime

        return f"{commit} (file terbaru {datetime.fromtimestamp(terbaru):%d/%m %H:%M})"
    except Exception:
        return commit


# --- Satu instance saja ------------------------------------------------------


_kunci = None  # dipegang selama proses hidup; jangan sampai kena garbage collect


def klaim_satu_instance() -> int | None:
    """Pastiin cuma ada SATU agent. Balikin PID pemegang lama kalau gagal.

    Kenapa ditegakkan di kode, bukan diserahin ke kebiasaan: agent nyangkut
    hotkey global. Dua agent artinya tiap pencetan ditangkep dua-duanya, dan
    keduanya rebutan mic. Ini pernah kejadian beneran — autostart nyalain satu
    pas login, terus satu lagi dijalanin manual buat ngetes, dan mode sesi
    kelihatan kayak nggak jalan padahal yang nyaut agent lama.

    Pakai kunci file dari OS, bukan sekadar file PID: kunci OS dilepas otomatis
    pas proses mati, jadi agent yang crash nggak ninggalin file yang bikin agent
    berikutnya nolak jalan selamanya.
    """
    global _kunci
    path = config.MEMORY_DIR / "agent.lock"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import msvcrt
    except ImportError:  # bukan Windows — lewat, jangan bikin agent gagal jalan
        log.debug("kunci satu-instance dilewat: msvcrt nggak ada")
        return None

    f = open(path, "a+")
    try:
        f.seek(0)
        # Kunci byte 0 doang. PID ditulis MULAI byte 1, jadi nulis PID nggak
        # nyentuh byte yang dikunci.
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.seek(1)
        lama = f.read().strip()
        f.close()
        try:
            return int(lama)
        except ValueError:
            return 0  # ada yang megang, tapi PID-nya nggak kebaca

    f.seek(1)
    f.truncate(1)
    f.write(str(os.getpid()))
    f.flush()
    _kunci = f  # ditahan sengaja: file ketutup = kunci lepas
    return None


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
            "sesi" if config.SESSION_MODE else config.HOTKEY_MODE,
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
            "sesi" if config.SESSION_MODE else config.HOTKEY_MODE,
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

# Memory-wipe phrases. Matched locally, not through the LLM: an irreversible
# command must not hinge on a model's guess.
_FORGET_PHRASES = (
    "forget everything",
    "forget it all",
    "erase your memory",
    "clear your memory",
    "wipe your memory",
    "delete your memory",
    "forget what you know about me",
)


def _wants_forget(text: str) -> bool:
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return any(f in clean for f in _FORGET_PHRASES)


# An event read back but not yet saved. Confirmation is deliberate: a mishearing
# on a write leaves a bogus entry that only surfaces next week.
_pending_event: dict | None = None


# Tugas yang lagi dikumpulin datanya, belum disimpen.
#
# Dulu tugas langsung disimpen apa pun isinya, dan hasilnya entri kayak
#   {'title': 'Assignment', 'due': '2026-08-19', 'course': '', 'estimate_hours': 0}
# dari ucapan "add some assignment for uh fourteen deadline" — judul generik,
# tenggat SALAH (dibilang 14, kesimpen 19), matkul & lama ngerjain kosong.
# Entri kayak gitu nggak bisa dipakai mutusin apa pun, dan kesimpen diem-diem
# tanpa dibacain ulang.
_pending_task: dict | None = None


def _tanggal_ucap(iso: str) -> str:
    d = datetime.strptime(iso, "%Y-%m-%d").date()
    return f"{tugas.DAYS[d.weekday()]} {tugas.MONTHS[d.month]} {d.day}"


def _bacain_tugas(t: dict) -> str:
    jam = t["estimate_hours"]
    jam_ucap = f"{jam:g} hour" + ("" if jam == 1 else "s")
    return (
        f"{t['title']} for {t['course']}, due {_tanggal_ucap(t['due'])}, "
        f"about {jam_ucap}. Save it?"
    )


def _add_task(text: str) -> None:
    """Kumpulin dulu sampai lengkap, baru bacain, baru simpen.

    Beda dari sebelumnya yang langsung nyimpen apa adanya. Tugas setengah isi
    itu lebih buruk daripada nggak ada tugas: dia nongol di daftar, keitung pas
    ditanya "hari ini ngerjain apa", tapi nggak bisa dipakai mutusin apa-apa.
    """
    global _pending_task
    try:
        t = jadwal_baru.parse_task(text, llm.get_conversation()._oneshot)
    except Exception:
        log.exception("failed to parse task")
        _say_safely("Sorry, I didn't catch the task.")
        return

    if t is None:
        _say_safely("Sorry, that task wasn't clear. Could you say it again?")
        return

    # Pengurai deterministik nimpa model buat field yang bisa diurai pasti —
    # alasan yang sama kayak time_en di acara kalender.
    t.update(tugas.ekstrak(text))
    _pending_task = t
    _lanjutin_tugas()


def _lanjutin_tugas() -> None:
    """Tanyain yang kurang, atau bacain kalau udah lengkap."""
    t = _pending_task
    if t is None:
        return
    hilang = tugas.kurang(t)
    if hilang:
        log.info("tugas belum lengkap, kurang %s: %r", hilang, t)
        _say_safely(tugas.kalimat_kurang(hilang))
        return
    log.info("tugas nunggu konfirmasi: %r", t)
    _say_safely(_bacain_tugas(t))


def _handle_task_followup(text: str) -> bool:
    """Jawaban buat tugas yang lagi dikumpulin. True = udah ditangani.

    Dicek sebelum niat lain, karena "ENGN4122" atau "about eight hours" itu
    nggak kelihatan kayak niat apa pun — tanpa ini jawabannya nyasar ke model.
    """
    global _pending_task
    if _pending_task is None:
        return False

    jawab = jadwal_baru.answer_yes(text)

    # Batalin duluan: "cancel" harus menang walau kalimatnya juga ngandung
    # potongan yang bisa diurai.
    if jawab is False:
        log.info("tugas dibatalin user")
        _pending_task = None
        _say_safely("Okay, dropped it.")
        return True

    lengkap = not tugas.kurang(_pending_task)
    if lengkap and jawab is True:
        t = _pending_task
        _pending_task = None
        tugas.add(t["title"], t["due"], t["course"], t["estimate_hours"])
        _say_safely(f"Saved. {t['title']} due {_tanggal_ucap(t['due'])}.")
        return True

    # Bukan ya/nggak — berarti isian buat field yang kurang.
    baru = tugas.ekstrak(text)
    if not baru:
        if lengkap:
            # Udah dibacain tapi jawabannya nggak jelas. Jangan nebak: nyimpen
            # tugas yang nggak disetujui itu arah yang mahal.
            _pending_task = None
            _say_safely("I wasn't sure what you said, so I didn't save it.")
            return True
        _say_safely("Sorry, I didn't catch that. " + tugas.kalimat_kurang(tugas.kurang(_pending_task)))
        return True

    _pending_task.update(baru)
    log.info("tugas dapet tambahan %r", baru)
    _lanjutin_tugas()
    return True


def _mark_task_done(text: str) -> None:
    t = tugas.mark(text, done=True)
    if t is None:
        # Nothing matched, or two matched — ask rather than guess which one
        _say_safely("Which task do you mean? I'm not sure.")
        return
    left = len(tugas.all_tasks())
    msg = f"Done, I've marked {t['title']} complete."
    msg += " That's everything." if left == 0 else f" {left} left."
    _say_safely(msg)


def _start_event(text: str) -> None:
    """Parse the utterance into an event, then read it back for confirmation."""
    global _pending_event
    try:
        event = jadwal_baru.parse_event(text, llm.get_conversation()._oneshot)
    except Exception:
        log.exception("failed to parse event")
        _say_safely("Sorry, I didn't catch the event details.")
        return

    if event is None or not event["confident"]:
        log.info("event details unclear: %r", event)
        _say_safely("Sorry, the date or time wasn't clear. Could you say it in full?")
        return

    _pending_event = event
    log.info("event awaiting confirmation: %r", event)
    _say_safely(jadwal_baru.confirmation_line(event))


def _handle_confirmation(text: str) -> bool:
    """True when this utterance was consumed as a confirmation answer."""
    global _pending_event
    if _pending_event is None:
        return False

    answer = jadwal_baru.answer_yes(text)
    if answer is None:
        # Neither yes nor no — drop it rather than guess. Guessing wrong here
        # writes an event the user never asked for.
        _pending_event = None
        log.info("confirmation unclear, event discarded")
        _say_safely("I wasn't sure what you said, so I didn't save it.")
        return True

    event = _pending_event
    _pending_event = None

    if not answer:
        log.info("event cancelled by user")
        _say_safely("Okay, cancelled.")
        return True

    # Local calendar takes precedence; Google only when local is off
    target = kalender_lokal if kalender_lokal.aktif() else gcal
    try:
        target.bikin_acara(
            event["title"], event["start"], event["end"], event["location"]
        )
    except Exception:
        log.exception("failed to create event (%s)", target.__name__)
        _say_safely("Sorry, I couldn't save that to your calendar.")
        return True

    calendar.refresh(paksa=True)  # so the agenda shows it right away
    _say_safely("Saved to your calendar.")
    return True


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
        log.info("model still loading, telling the user first")
        _say_safely("One moment, just getting ready.")

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
    _route_and_reply(text)


def _route_and_reply(text: str) -> None:
    """Percabangan niat + jawaban. Dipakai bareng mode pencet dan mode sesi.

    Urutannya PENTING dan nggak boleh diubah — lihat komentar tiap cabang.
    """
    # a. Memory wipe — handled locally, never sent to the LLM
    if _wants_forget(text):
        llm.get_conversation().forget()
        _say_safely("Okay, I've erased everything I knew about you.")
        return

    # b. Waiting on an event confirmation? This answer belongs to that.
    if _handle_confirmation(text):
        return

    # b2. Lagi ngumpulin data tugas? Jawabannya milik itu. Harus sebelum niat
    #     lain: "ENGN4122" atau "about eight hours" nggak kelihatan kayak niat
    #     apa pun, jadi tanpa ini jawabanmu nyasar ke model.
    if _handle_task_followup(text):
        return

    # c. Tasks. MUST be checked before the event intent: "add a task ..." also
    #    matches the "add ..." pattern, and taking the wrong branch turns the
    #    user's task into a calendar event.
    if tugas.wants_mark_done(text):
        _mark_task_done(text)
        return
    if tugas.wants_add_task(text):
        _add_task(text)
        return

    # d. Create a new event
    if (kalender_lokal.aktif() or gcal.aktif()) and jadwal_baru.wants_event(text):
        _start_event(text)
        return

    # e. Pertanyaan berjawaban tertutup — jam, tanggal, jadwal — dijawab dari
    #    hitungan Python, nggak lewat model sama sekali. Terukur: model salah
    #    baca jam 4/6 sampai 6/6 (dia MEMBULATKAN 20:12 jadi "quarter past
    #    eight"), sesekali nyebut kelas hari ini sebagai "tomorrow", dan
    #    nambah 3-12 detik buat data yang udah pasti.
    #
    #    answer() balikin None kalau ragu, jadi pertanyaan yang nggak dikenali
    #    tetep jatuh ke model. Prinsipnya sama kayak time_en.py: yang tertutup
    #    dikerjain kode, yang butuh pemahaman bahasa dikerjain model.
    pasti = jawab_pasti.answer(text)
    if pasti:
        log.info("dijawab tanpa LLM: %s", pasti)
        _ucap_kalimat(pasti)
        return

    # f. Sisanya ke LLM, ngomong kalimat per kalimat
    _speak_streaming(text)


def _speak_streaming(text: str) -> None:
    """Jawab sambil ngomong: kalimat pertama dibunyiin selagi model masih
    ngarang kalimat berikutnya.

    Terukur: waktu sampai bunyi pertama turun 53-71% dibanding nunggu seluruh
    jawaban jadi. Yang dipangkas bukan waktu totalnya, tapi diamnya — dan itu
    yang bikin percakapan kerasa hidup atau kerasa nge-lag.
    """
    conv = llm.get_conversation()
    sp = audio.Speaker()
    ada = False
    try:
        for kalimat in conv.chat_stream(text):
            sp.add(tts.speak(kalimat))
            ada = True
        sp.finish()
    except Exception:
        # Yang udah kebunyiin biarin selesai; jangan motong di tengah kata.
        try:
            sp.finish()
        except Exception:
            log.debug("gagal nutup playback", exc_info=True)
        log.exception("gagal jawab (%s)", config.LLM_BACKEND)
        if not ada:
            _say_safely("Sorry, my brain isn't reachable right now.")


def handle_session(is_active: Callable[[], bool]) -> None:
    """Mode sesi: sekali pencet hotkey, terus ngobrol tanpa mencet lagi.

    Batas kalimat ditentuin VAD, bukan tombol. Sesi tutup kalau kamu mencet
    hotkey lagi ATAU diam selama SESSION_IDLE_SECONDS.

    Mic ditutup selama agent ngomong — itu konsekuensi sadar dari nggak ada
    barge-in. Untungnya: agent nggak mungkin denger suaranya sendiri, jadi
    nggak perlu peredam gema dan speaker biasa aman dipakai.
    """
    if not stt.is_loaded():
        threading.Thread(target=stt.warmup, name="muat-duluan", daemon=True).start()

    audio.beep_start()
    log.info("sesi dibuka (tutup: hotkey lagi, atau diam %.0f detik)",
             config.SESSION_IDLE_SECONDS)
    giliran = 0
    mulai = time.monotonic()

    try:
        while is_active():
            try:
                clip, alasan = vad.rekam_ucapan(berhenti=lambda: not is_active())
            except Exception:
                log.exception("gagal merekam mic")
                audio.beep_error()
                return

            if clip is None:
                log.info("sesi ditutup: %s", alasan)
                break

            if not stt.is_loaded():
                log.info("model masih dimuat, kabarin user dulu")
                _say_safely("One moment, just getting ready.")

            try:
                text = stt.transcribe(clip)
            except Exception:
                log.exception("gagal transcribe")
                audio.beep_error()
                continue  # satu ucapan gagal nggak boleh nutup sesi

            if not text:
                log.info("transcript kosong, lanjut nunggu")
                continue

            giliran += 1
            log.info("User (giliran %d): %s", giliran, text)

            if _wants_end_session(text):
                log.info("sesi ditutup lewat ucapan")
                _say_safely("Okay, talk to you later.")
                break

            try:
                _route_and_reply(text)
            except Exception:
                # Satu giliran yang gagal jangan ngebunuh sesinya — user tinggal
                # ngomong lagi. Yang fatal cuma mic-nya yang nggak bisa dibuka.
                log.exception("giliran %d gagal", giliran)
                audio.beep_error()
    finally:
        audio.beep_stop()
        log.info("sesi selesai: %d giliran, %.0f detik", giliran,
                 time.monotonic() - mulai)


# Frasa penutup sesi. Dicocokin lokal, bukan lewat LLM — sama alasannya kayak
# "forget everything": perintah kontrol nggak boleh gantung pada tebakan model.
# Dua tingkat, dan pemisahannya penting.
#
# KUAT = nggak mungkin punya arti lain, jadi boleh cocok di mana pun kalimat.
_END_STRONG = (
    "goodbye", "good bye", "bye bye", "talk to you later", "see you later",
    "stop listening", "end session", "end the session",
)
# LEMAH = cuma nutup kalau berdiri di UJUNG ucapan. "I'm done" nutup sesi, tapi
# "I'm done with assignment one" itu laporan tugas selesai — kalau dicocokin di
# mana pun, tugasmu nggak akan pernah ketandai karena sesinya keburu tutup.
_END_WEAK = (
    "bye", "stop", "thanks", "thank you", "see you",
    "thats all", "thats it", "were done", "im done", "we are done", "i am done",
)


def _wants_end_session(text: str) -> bool:
    t = teks.normal(text)
    if not t:
        return False
    if any(f in t for f in _END_STRONG):
        return True
    return any(t == f or t.endswith(" " + f) for f in _END_WEAK)


def _ucap_kalimat(text: str) -> None:
    """Ucapkan per kalimat, mulai bunyi begitu kalimat pertama jadi.

    Alasannya sama kayak jalur streaming: TTS baru bisa mulai setelah kalimat
    utuh, jadi nyintesis seluruh jawaban dulu bikin diamnya kepanjangan.
    Jawaban pasti sempat lebih lambat dari model persis gara-gara ini.
    """
    from .llm import _potong_kalimat

    sp = audio.Speaker()
    try:
        # Spasi di ujung wajib: _potong_kalimat() nyari tanda baca yang DIIKUTI
        # spasi, jadi kalimat terakhir nggak bakal kedeteksi tanpa ini.
        sisa = text.rstrip() + " "
        ada = False
        while True:
            kal, sisa = _potong_kalimat(sisa)
            if kal is None:
                break
            sp.add(tts.speak(kal))
            ada = True
        ekor = sisa.strip()
        if ekor:
            sp.add(tts.speak(ekor))
            ada = True
        if not ada:
            sp.add(tts.speak(text))
        sp.finish()
    except Exception:
        try:
            sp.finish()
        except Exception:
            log.debug("gagal nutup playback", exc_info=True)
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

    TTS selalu dimuat: CPU-only, 0 VRAM, cuma ~1,5-2,7 detik. STT tergantung
    WHISPER_WARMUP — kalau modelnya toh bakal dilepas pas nganggur, muat di awal
    cuma nahan memori percuma sampai ambang idle kelewat.

    VAD ikut dimuat cuma di mode sesi, dan selalu: 0 VRAM, ~1 detik, tapi kalau
    baru dimuat pas hotkey dipencet, kata pertamamu kepotong.
    """

    def _run():
        try:
            tts.get_voice()
            if config.SESSION_MODE:
                vad._model()
            if config.WHISPER_WARMUP:
                stt.warmup()
            else:
                log.info(
                    "STT (%s) nggak di-warmup; dimuat pas pertanyaan pertama",
                    config.STT_BACKEND,
                )
            log.info("Warmup selesai, siap dipakai.")
        except Exception:
            log.exception("warmup gagal (model bakal di-load pas dipakai)")

    threading.Thread(target=_run, name="warmup", daemon=True).start()


def main() -> int:
    setup_logging()
    log.info("=" * 60)
    log.info("build: %s", _versi_build())
    otak = (
        config.OLLAMA_MODEL if config.LLM_BACKEND == "ollama" else config.CLAUDE_MODEL
    )
    log.info(
        "personal-agent start | lang=%s offline=%s | hotkey=%s (%s, %s) | "
        "stt=%s/%s | llm=%s/%s | tts=%s",
        config.LANGUAGE,
        config.OFFLINE_MODE,
        config.HOTKEY,
        "SESI" if config.SESSION_MODE else config.HOTKEY_MODE,
        config.HOTKEY_BACKEND,
        config.STT_BACKEND,
        config.STT_DEVICE,
        config.LLM_BACKEND,
        otak,
        config.TTS_BACKEND,
    )

    # Sebelum apa-apa: pastiin nggak ada agent lain. Dicek DULUAN karena agent
    # kedua yang terlanjur muat model bakal makan memori percuma sebelum nyerah.
    lain = klaim_satu_instance()
    if lain is not None:
        log.error(
            "Agent lain udah jalan (PID %s). Yang ini berhenti — dua agent bakal "
            "rebutan hotkey '%s'. Cek: powershell -File scripts\status.ps1",
            lain or "?",
            config.HOTKEY,
        )
        return 3

    # Setelan yang bentrok sama mode offline harus ketahuan di log startup,
    # bukan pas user udah nanya.
    try:
        config.wajib_offline()
    except config.OfflineViolation:
        log.exception("setelan bentrok sama OFFLINE_MODE, agent nggak jalan")
        return 2

    warmup()
    # Mode sesi ganti seluruh gaya interaksinya, bukan sekadar setelan:
    # hotkey jadi pembuka sesi, bukan tombol rekam.
    backend = make_backend(handle_session if config.SESSION_MODE else handle_utterance)

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
