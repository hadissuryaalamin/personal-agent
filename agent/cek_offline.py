"""Cek kesiapan jalan offline: setelan, bobot model, dan Ollama.

Dipisah dari agent supaya bisa dijalanin kapan aja tanpa nyalain hotkey
listener — dan supaya kegagalan ketahuan di terminal, bukan cuma di log file
yang jarang dibuka.

    .venv-agent\\Scripts\\python.exe -m agent.cek_offline
"""

import sys
from pathlib import Path

from . import config

OK = "  ok    "
GAGAL = "  GAGAL "
INFO = "  --    "


def _setelan() -> list[str]:
    print("[1/4] Setelan")
    masalah = config._cek_offline()
    if not config.OFFLINE_MODE:
        print(f"{INFO}OFFLINE_MODE=false, jadi backend jaringan boleh dipakai")
        return []
    if masalah:
        for m in masalah:
            print(f"{GAGAL}{m}")
    else:
        print(f"{OK}semua backend lokal")
    return masalah


def _berkas() -> list[str]:
    print("\n[2/4] Bobot model di disk")
    masalah = []
    perlu: list[tuple[str, Path]] = []
    if config.TTS_BACKEND == "kokoro":
        perlu += [("Kokoro model", config.KOKORO_MODEL),
                  ("Kokoro voices", config.KOKORO_VOICES)]
    elif config.TTS_BACKEND == "piper":
        perlu.append(("Piper voice", Path(config.PIPER_VOICE)))

    for nama, p in perlu:
        p = Path(p)
        if p.exists() and p.stat().st_size > 0:
            print(f"{OK}{nama}: {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
        else:
            masalah.append(f"{nama} nggak ada di {p}")
            print(f"{GAGAL}{nama} nggak ada di {p}")
    return masalah


def _model_dimuat() -> list[str]:
    """Muat STT & TTS beneran. Satu-satunya cara tau bobotnya lengkap — file
    ada tapi korup itu kelihatan sama aja dari luar."""
    print("\n[3/4] Model kemuat (unduh kalau belum ada — butuh jaringan sekali)")
    masalah = []
    import time

    for nama, muat in (("STT", _muat_stt), ("TTS", _muat_tts)):
        t0 = time.perf_counter()
        try:
            muat()
            print(f"{OK}{nama} kemuat ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:
            masalah.append(f"{nama} gagal dimuat: {e}")
            print(f"{GAGAL}{nama} gagal dimuat: {e}")
    return masalah


def _muat_stt() -> None:
    from . import stt

    stt.get_model()


def _muat_tts() -> None:
    from . import tts

    if not tts.speak("ready"):
        raise RuntimeError("nggak ngasilin audio")


def _otak() -> list[str]:
    print("\n[4/4] Otak")
    if config.LLM_BACKEND != "ollama":
        print(f"{INFO}LLM_BACKEND={config.LLM_BACKEND}, lewati cek Ollama")
        return []

    import requests

    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        punya = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print(f"{GAGAL}Ollama nggak nyaut di {config.OLLAMA_URL}: {e}")
        return [f"Ollama nggak jalan di {config.OLLAMA_URL}"]

    # Ollama nulis 'qwen2.5:7b'; setelan boleh nulis tanpa tag.
    mau = config.OLLAMA_MODEL
    if any(n == mau or n.split(":")[0] == mau.split(":")[0] for n in punya):
        print(f"{OK}Ollama jalan, {mau} ada")
        return []
    print(f"{GAGAL}{mau} belum di-pull. Yang ada: {', '.join(punya) or '(kosong)'}")
    return [f"model {mau} belum di-pull"]


def main() -> int:
    print("=== cek kesiapan offline ===")
    print(f"lang={config.LANGUAGE} offline={config.OFFLINE_MODE} | "
          f"stt={config.STT_BACKEND} llm={config.LLM_BACKEND} tts={config.TTS_BACKEND}\n")

    masalah = _setelan() + _berkas() + _model_dimuat() + _otak()

    print()
    if masalah:
        print(f"{len(masalah)} masalah:")
        for m in masalah:
            print(f"  - {m}")
        return 1
    print("Semua siap. Agent bisa jalan tanpa jaringan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
