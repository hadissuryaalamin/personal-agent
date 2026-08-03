"""Baca jadwal dari link ICS (read-only).

Sengaja lewat ICS, bukan API: nggak perlu OAuth, nggak perlu bikin project di
cloud console, dan agent nggak punya izin nulis ke kalendermu. Konsekuensinya
memang nggak bisa bikin acara — itu butuh API sungguhan.

Hasilnya diselipin ke system prompt sebagai teks agenda, jadi agent bisa jawab
"besok ada apa" tanpa perlu ronde tool-call tambahan (yang bakal nambah jeda).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: list[dict] | None = None
_cache_time = 0.0
_refresh_jalan = False

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Kode jenis kelas di feed ANU: LecA, TutA, ComA, ...
KINDS = {
    "lec": "lecture",
    "tut": "tutorial",
    "com": "computer lab",
    "lab": "lab",
    "wor": "workshop",
    "sem": "seminar",
    "dro": "drop-in",
}

# Judul yang ditulis sebelum pindah ke Inggris masih bawa label Indonesia.
# Tetap dikenali biar file .ics lama nggak jadi '(kuliah)' bocor ke prompt —
# lihat ARSITEKTUR.md soal kenapa data lama nggak boleh bikin parser gagal.
LEGACY_KINDS = {
    "kuliah": "lecture",
    "tutorial": "tutorial",
    "lab komputer": "computer lab",
    "lab": "lab",
    "workshop": "workshop",
    "seminar": "seminar",
    "sesi konsultasi": "drop-in",
}


def _kind_names() -> set[str]:
    """Semua bentuk label jenis yang boleh muncul di dalam kurung."""
    return set(KINDS.values()) | set(LEGACY_KINDS)


def _cache_path():
    return config.MEMORY_DIR / "calendar.ics"


def _bersihin_judul(summary: str) -> str:
    """'Advanced Topics in Human-Centr_Agentic Coding Studio (Class: 9056), LecA'
    -> 'Agentic Coding Studio'

    Format ANU: JudulUmum_NamaSpesifik. Yang umum ('Advanced Topics in AI')
    dibuang karena kode matkulnya udah nunjukin itu, dan ANU sering motong
    judulnya di tengah ('...Human-Centr'). Nomor kelas & kode jenis juga
    dibuang — nggak ada gunanya pas dibacakan.
    """
    teks = re.sub(r"\(Class:\s*\d+\)", "", summary)
    # Backslash-nya buat jaga-jaga: kalau teksnya belum lewat pelepas escape
    # ICS, koma pemisah masih ketulis '\,'
    teks = re.sub(r"\\?,\s*(Lec|Tut|Com|Lab|Wor|Sem|Dro)\w*\s*$", "", teks.strip())
    # Buang '(kuliah)' dsb yang nempel di judul acara hasil salinan — tapi cuma
    # kalau isinya memang nama jenis kelas, biar 'Makan siang (sama Budi)'
    # nggak ikut terpotong.
    m = re.match(r"^(.*)\(([^)]+)\)\s*$", teks)
    if m and m.group(2).strip().lower() in _kind_names():
        teks = m.group(1)
    if "_" in teks:
        bagian = [b.strip() for b in teks.split("_") if b.strip()]
        if bagian:
            teks = bagian[-1]
    return re.sub(r"[\s,]+$", "", re.sub(r"\s+", " ", teks)).strip()


def _jenis_kelas(summary: str) -> str:
    m = re.search(r"\\?,\s*(Lec|Tut|Com|Lab|Wor|Sem|Dro)\w*\s*$", summary.strip())
    if m:
        return KINDS.get(m.group(1).lower(), "")
    # Bentuk kedua: '... (kuliah)' — muncul di acara yang pernah disalin dari
    # feed ANU ke Google atau ke file lokal, di mana jenisnya ikut ke judul.
    m = re.match(r"^.*\(([^)]+)\)\s*$", summary.strip())
    if m:
        isi = m.group(1).strip().lower()
        # Label lama diterjemahkan di sini, bukan disimpan apa adanya, biar
        # yang sampai ke prompt selalu satu bahasa.
        if isi in LEGACY_KINDS:
            return LEGACY_KINDS[isi]
        if isi in set(KINDS.values()):
            return isi
    return ""


def _kode_matkul(description: str) -> str:
    m = re.match(r"\s*([A-Z]{4}\d{4})", description or "")
    return m.group(1) if m else ""


def _bersihin_lokasi(lokasi: str) -> str:
    """'Rm 2.02_Fulton Muir Bldg 95' -> 'Fulton Muir, Rm 2.02'

    Nama gedung didahulukan karena itu yang dicari orang duluan; nomor gedung
    dibuang karena nggak nolong pas didengar.
    """
    if not lokasi:
        return ""
    teks = re.sub(r"\s*Bldg\s*\d+\s*", "", lokasi).strip()
    bagian = [b.strip() for b in teks.split("_") if b.strip()]
    if len(bagian) == 2 and bagian[0].lower().startswith(("rm", "room")):
        bagian = [bagian[1], bagian[0]]
    return ", ".join(bagian)


def _ambil_ics() -> str:
    import requests

    resp = requests.get(config.CALENDAR_ICS_URL, timeout=config.CALENDAR_TIMEOUT)
    resp.raise_for_status()
    # Feed ANU nggak nyantumin charset, jadi requests nebak ISO-8859-1 dan
    # bikin apostrof jadi 'â€™'. ICS itu UTF-8 menurut RFC 5545.
    return resp.content.decode("utf-8")


def _parse(teks: str) -> list[dict]:
    from icalendar import Calendar

    tz = ZoneInfo(config.CALENDAR_TZ)
    hasil = []
    for komponen in Calendar.from_ical(teks).walk("VEVENT"):
        mulai = komponen.get("DTSTART")
        if mulai is None:
            continue
        mulai = mulai.dt
        selesai = komponen.get("DTEND")
        selesai = selesai.dt if selesai is not None else None

        # Acara seharian datang sebagai date, bukan datetime
        sepanjang_hari = not isinstance(mulai, datetime)
        if not sepanjang_hari:
            mulai = mulai.astimezone(tz)
            if isinstance(selesai, datetime):
                selesai = selesai.astimezone(tz)

        summary = str(komponen.get("SUMMARY", "")).strip()
        desc = str(komponen.get("DESCRIPTION", "")).strip()
        hasil.append(
            {
                "mulai": mulai,
                "selesai": selesai,
                "sepanjang_hari": sepanjang_hari,
                "judul": _bersihin_judul(summary),
                "jenis": _jenis_kelas(summary),
                "kode": _kode_matkul(desc),
                "lokasi": _bersihin_lokasi(str(komponen.get("LOCATION", "")).strip()),
            }
        )
    hasil.sort(key=lambda e: (e["mulai"].isoformat() if not e["sepanjang_hari"] else str(e["mulai"])))
    log.info("kalender: %d acara ke-parse", len(hasil))
    return hasil


def _muat_cache_disk() -> list[dict] | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        return _parse(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("cache kalender di disk rusak", exc_info=True)
        return None


def refresh(paksa: bool = False) -> None:
    """Tarik ulang dari server. Aman dipanggil dari thread mana pun."""
    global _cache, _cache_time
    if not config.CALENDAR_ICS_URL:
        return
    umur = time.monotonic() - _cache_time
    if not paksa and _cache is not None and umur < config.CALENDAR_CACHE_MINUTES * 60:
        return
    try:
        teks = _ambil_ics()
        acara = _parse(teks)
        with _lock:
            _cache = acara
            _cache_time = time.monotonic()
        # Simpan biar restart nggak perlu nunggu jaringan, dan tetep jalan offline
        config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path().write_text(teks, encoding="utf-8")
    except Exception:
        log.warning("gagal ambil kalender (pakai cache lama kalau ada)", exc_info=True)


def _refresh_latar() -> None:
    """Segarkan di latar belakang — jangan bikin user nunggu jaringan."""
    global _refresh_jalan
    with _lock:
        if _refresh_jalan:
            return
        _refresh_jalan = True

    def _run():
        global _refresh_jalan
        try:
            refresh()
        finally:
            _refresh_jalan = False

    threading.Thread(target=_run, name="kalender", daemon=True).start()


def _acara() -> list[dict]:
    """Acara dari sumber berbasis ICS: file lokal atau feed URL."""
    global _cache, _cache_time

    # File lokal nggak perlu cache/jaringan — baca langsung tiap kali biar
    # acara yang baru ditulis lewat suara langsung kelihatan.
    from . import kalender_lokal

    if kalender_lokal.aktif():
        try:
            return _parse(kalender_lokal.teks())
        except Exception:
            log.warning("kalender lokal nggak keparse", exc_info=True)
            return []

    # Wajib dicek: cache di disk tetep ada walau URL-nya dikosongin. Tanpa ini,
    # acara yang udah dipindah ke Google kebaca dua kali — sekali dari cache
    # basi, sekali dari Google.
    if not config.CALENDAR_ICS_URL:
        return []
    if _cache is None:
        with _lock:
            if _cache is None:
                dari_disk = _muat_cache_disk()
                if dari_disk is not None:
                    _cache = dari_disk
                    _cache_time = 0.0  # tetep picu penyegaran
    if _cache is None:
        refresh(paksa=True)  # belum ada apa-apa: terpaksa nunggu
    else:
        _refresh_latar()
    return _cache or []


def _label_hari(d: date, hari_ini: date) -> str:
    beda = (d - hari_ini).days
    nama = f"{DAYS[d.weekday()]} {d.day} {MONTHS[d.month]}"
    if beda == 0:
        return f"Today ({nama})"
    if beda == 1:
        return f"Tomorrow ({nama})"
    if beda == 2:
        return f"Day after tomorrow ({nama})"
    return nama


def agenda() -> str:
    """Teks jadwal buat diselipin ke system prompt. Kosong kalau nggak ada sumber.

    Dua sumber, masing-masing opsional: feed ICS dan Google Calendar. Cukup
    salah satu aktif.
    """
    from . import gcal, kalender_lokal

    if (
        not config.CALENDAR_ICS_URL
        and not gcal.aktif()
        and not kalender_lokal.aktif()
    ):
        return ""

    tz = ZoneInfo(config.CALENDAR_TZ)
    sekarang = datetime.now(tz)
    hari_ini = sekarang.date()
    batas = hari_ini + timedelta(days=config.CALENDAR_DAYS_AHEAD)

    semua = list(_acara())

    # Gabung acara pribadi dari Google. Kalau jadwal ANU juga udah diimpor ke
    # Google, kosongin CALENDAR_ICS_URL — kalau nggak, tiap kelas kebaca dua kali.
    from . import gcal

    jangkauan = max(
        config.CALENDAR_DAYS_AHEAD + 1, config.CALENDAR_LOOKAHEAD_DAYS
    )
    if gcal.aktif():
        try:
            semua += gcal.ambil_acara(jangkauan)
        except Exception:
            log.warning("gagal ambil acara Google", exc_info=True)

    semua.sort(
        key=lambda e: (
            e["mulai"].isoformat() if not e["sepanjang_hari"] else str(e["mulai"])
        )
    )

    per_hari: dict[date, list[dict]] = {}
    for e in semua:
        d = e["mulai"].date() if not e["sepanjang_hari"] else e["mulai"]
        if hari_ini <= d < batas:
            per_hari.setdefault(d, []).append(e)

    if not per_hari:
        return (
            f"User's schedule for the next {config.CALENDAR_DAYS_AHEAD} days: "
            "nothing, no classes."
        )

    baris = [f"User's schedule for the next {config.CALENDAR_DAYS_AHEAD} days."]

    # Hitung "kelas berikutnya" di sini, jangan diserahin ke model. Nyari acara
    # terdekat dari sekarang itu penalaran lintas-baris — persis hal yang bikin
    # model salah ambil data dari baris tetangga.
    berikutnya = None
    for e in semua:
        if e["sepanjang_hari"]:
            continue
        if e["mulai"] > sekarang:
            berikutnya = e
            break
    if berikutnya is not None:
        baris.append("NEXT UP: " + _satu_baris(berikutnya, hari_ini))

    # Dua pertanyaan paling sering ("hari ini apa", "besok apa") dijawab di sini
    # juga, dalam bentuk jadi. Model kecil sering salah hari atau cuma nyebut
    # satu dari beberapa kelas kalau disuruh baca tabelnya sendiri.
    for offset, label in ((0, "TODAY"), (1, "TOMORROW")):
        d = hari_ini + timedelta(days=offset)
        isi = per_hari.get(d, [])
        when = f"{DAYS[d.weekday()]} {MONTHS[d.month]} {d.day}"
        if not isi:
            baris.append(f"{label} ({when}): nothing scheduled")
        else:
            ringkas = "; ".join(
                _satu_baris(e, hari_ini, dengan_hari=False) for e in isi
            )
            baris.append(f"{label} ({when}): {len(isi)} scheduled -> {ringkas}")

    baris.append("")
    baris.append("Each line: time | code & name | kind | location")
    for d in sorted(per_hari):
        baris.append(_label_hari(d, hari_ini) + ":")
        for e in per_hari[d]:
            baris.append("  " + _satu_baris(e, hari_ini, dengan_hari=False))

    luar = _di_luar_pola(semua, per_hari, hari_ini, batas)
    if luar:
        baris.append("")
        baris.append(
            f"Beyond those {config.CALENDAR_DAYS_AHEAD} days, only what breaks the "
            "weekly pattern (routine classes are not repeated here):"
        )
        baris.extend("  " + b for b in luar)
    return "\n".join(baris)


def _sidik(e: dict) -> tuple:
    """Ciri acara buat ngenali pola mingguan: hari + jam + judul."""
    if e["sepanjang_hari"]:
        return ("harian", e["judul"])
    return (e["mulai"].weekday(), e["mulai"].strftime("%H:%M"), e["judul"])


def _di_luar_pola(
    semua: list[dict], per_hari: dict, hari_ini: date, batas: date
) -> list[str]:
    """Acara jauh di depan yang BUKAN pengulangan kelas mingguan.

    Jadwal kuliah berulang tiap minggu, jadi minggu kedua isinya sama persis
    dengan minggu pertama. Ngirim semuanya ke tiap permintaan itu bayar mahal
    buat informasi yang sama berulang-ulang. Yang beneran belum kelihatan dari
    jendela rinci cuma yang menyimpang: ujian, kelas pengganti, acara pribadi.
    """
    if config.CALENDAR_LOOKAHEAD_DAYS <= 0:
        return []

    rutin = {_sidik(e) for hari in per_hari.values() for e in hari}
    ujung = hari_ini + timedelta(days=config.CALENDAR_LOOKAHEAD_DAYS)

    hasil = []
    for e in semua:
        d = e["mulai"].date() if not e["sepanjang_hari"] else e["mulai"]
        if not (batas <= d < ujung):
            continue
        if _sidik(e) in rutin:
            continue
        rutin.add(_sidik(e))  # jangan diulang kalau dia sendiri berulang
        hasil.append(_satu_baris(e, hari_ini))
    return hasil


def _jam_ucap(e: dict) -> str:
    """Time in a form that reads aloud well, not '09:00-11:00'.

    The model copies whatever format it sees, and synthesisers spell out
    clock-formatted digits literally — measured 4.1s versus 3.0s for the same
    sentence. 24-hour is kept because "at 2" is ambiguous between day and night.
    """
    if e["sepanjang_hari"]:
        return "all day"

    def one(dt) -> str:
        return f"{dt.hour}" + (f" {dt.minute:02d}" if dt.minute else "")

    text = one(e["mulai"])
    if isinstance(e["selesai"], datetime):
        text += " to " + one(e["selesai"])
    return text


def _satu_baris(e: dict, hari_ini: date, dengan_hari: bool = True) -> str:
    """Satu acara jadi satu baris berkolom.

    Kolomnya dipisah '|' dengan sengaja: tanpa batas yang jelas, model gampang
    ngambil nama matkul dari baris ini tapi lokasinya dari baris sebelahnya.
    """
    jam = _jam_ucap(e)
    nama = f"{e['kode']} {e['judul']}".strip() if e["kode"] else e["judul"]
    kolom = [jam, nama, e["jenis"] or "-", e["lokasi"] or "location not given"]
    teks = " | ".join(kolom)

    if dengan_hari:
        d = e["mulai"].date() if not e["sepanjang_hari"] else e["mulai"]
        teks = f"{_label_hari(d, hari_ini)} | {teks}"
    return teks


def konteks_waktu() -> str:
    """Tanggal & jam sekarang. Model nggak punya jam — tanpa ini dia nggak bisa
    ngartiin 'besok' atau 'minggu depan'."""
    tz = ZoneInfo(config.CALENDAR_TZ)
    n = datetime.now(tz)
    return (
        f"Right now it is {DAYS[n.weekday()]}, {n.day} {MONTHS[n.month]} "
        f"{n.year}, {n.strftime('%H:%M')} in {config.CALENDAR_TZ}."
    )
