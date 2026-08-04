"""Check readiness to run offline: settings, model weights, and Ollama.

Kept separate from the agent so it can be run any time without starting the
hotkey listener — and so failures surface in the terminal rather than only in a
log file nobody opens.

    .venv-agent\\Scripts\\python.exe -m agent.offline_check
"""

import sys
from pathlib import Path

from . import config

OK = "  ok    "
FAIL = "  FAIL  "
INFO = "  --    "


def _settings() -> list[str]:
    print("[1/4] Settings")
    problems = config.offline_problems()
    if not config.OFFLINE_MODE:
        print(f"{INFO}OFFLINE_MODE=false, so network backends are allowed")
        return []
    if problems:
        for p in problems:
            print(f"{FAIL}{p}")
    else:
        print(f"{OK}all backends are local")
    return problems


def _files() -> list[str]:
    print("\n[2/4] Model weights on disk")
    problems = []
    needed: list[tuple[str, Path]] = []
    if config.TTS_BACKEND == "kokoro":
        needed += [("Kokoro model", config.KOKORO_MODEL),
                   ("Kokoro voices", config.KOKORO_VOICES)]
    elif config.TTS_BACKEND == "piper":
        needed.append(("Piper voice", Path(config.PIPER_VOICE)))

    for name, p in needed:
        p = Path(p)
        if p.exists() and p.stat().st_size > 0:
            print(f"{OK}{name}: {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
        else:
            problems.append(f"{name} missing at {p}")
            print(f"{FAIL}{name} missing at {p}")
    return problems


def _models_load() -> list[str]:
    """Actually load STT and TTS. The only way to know the weights are complete
    — a corrupt file looks identical to a good one from the outside."""
    print("\n[3/4] Models load (downloads if missing — needs the network once)")
    problems = []
    import time

    for name, load in (("STT", _load_stt), ("TTS", _load_tts)):
        t0 = time.perf_counter()
        try:
            load()
            print(f"{OK}{name} loaded ({time.perf_counter() - t0:.1f} s)")
        except Exception as e:
            problems.append(f"{name} failed to load: {e}")
            print(f"{FAIL}{name} failed to load: {e}")
    return problems


def _load_stt() -> None:
    from . import stt

    stt.get_model()


def _load_tts() -> None:
    from . import tts

    if not tts.speak("ready"):
        raise RuntimeError("produced no audio")


def _brain() -> list[str]:
    print("\n[4/4] Brain")
    if config.LLM_BACKEND != "ollama":
        print(f"{INFO}LLM_BACKEND={config.LLM_BACKEND}, skipping the Ollama check")
        return []

    import requests

    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        have = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print(f"{FAIL}Ollama did not answer at {config.OLLAMA_URL}: {e}")
        return [f"Ollama is not running at {config.OLLAMA_URL}"]

    # Ollama reports 'qwen2.5:7b'; the setting may omit the tag.
    want = config.OLLAMA_MODEL
    if any(n == want or n.split(":")[0] == want.split(":")[0] for n in have):
        print(f"{OK}Ollama is up, {want} is present")
        return []
    print(f"{FAIL}{want} has not been pulled. Available: {', '.join(have) or '(none)'}")
    return [f"model {want} has not been pulled"]


def main() -> int:
    print("=== offline readiness check ===")
    print(f"lang={config.LANGUAGE} offline={config.OFFLINE_MODE} | "
          f"stt={config.STT_BACKEND} llm={config.LLM_BACKEND} tts={config.TTS_BACKEND}\n")

    problems = _settings() + _files() + _models_load() + _brain()

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("All set. The agent can run with no network.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
