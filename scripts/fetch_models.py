"""One-time model download into ``models/``.

Invariant #8: nothing leaves the machine at runtime. This script is the only
thing in the repo that touches the network, it is not imported by anything in
``src/``, and it is meant to be run once by hand:

    python scripts\\fetch_models.py
    python scripts\\fetch_models.py --only llm
    python scripts\\fetch_models.py --check

Weights land in ``models/<name>/`` and are gitignored. Downloads resume, so a
dropped connection costs bandwidth rather than the whole file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

#: repo id -> (local directory, approximate size, what it is for)
TARGETS = {
    "llm": {
        "repo": "Qwen/Qwen3-4B-Instruct-2507",
        "dir": "qwen3-4b-instruct-2507",
        "gb": 8.1,
        "why": "the model whose hidden states the probe reads (M1-M3)",
        "allow": ["*.safetensors", "*.json", "*.txt", "*.py"],
    },
    "asr": {
        # The sherpa-onnx export of nvidia/parakeet-tdt-0.6b-v2. int8 because
        # sherpa-onnx runs this on the CPU while the LLM holds the GPU, and
        # PLAN.md section 5 gives ASR a 300 ms budget.
        "repo": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        "dir": "parakeet-tdt-0.6b-v2-int8",
        "gb": 0.7,
        "why": "speech to text (M4)",
        "allow": ["*.onnx", "tokens.txt", "test_wavs/*"],
    },
    "vad": {
        "repo": "deepghs/silero-vad-onnx",
        "dir": "silero-vad",
        "gb": 0.01,
        "why": "segments turns inside a session (M4)",
        "allow": ["silero_vad.onnx"],
    },
}

#: Not fetched from the network: Kokoro was downloaded once and lives in the
#: backup. fetch_models.py --restore-kokoro copies it back into models/.
KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
KOKORO_BACKUP = Path("E:/personal-agent-backup-20260806/models")


def target_path(key: str) -> Path:
    return MODELS / TARGETS[key]["dir"]


def free_gb(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1024**3


def check() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"models/ is on a volume with {free_gb(MODELS):.1f} GB free\n")
    missing = 0
    for key, spec in TARGETS.items():
        path = target_path(key)
        weights = (
            sorted(path.glob("*.safetensors")) + sorted(path.glob("*.onnx"))
            if path.exists()
            else []
        )
        if weights:
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3
            print(f"  {key:<5} present  {path.name}  ({size:.2f} GB, {len(weights)} files)")
        else:
            missing += 1
            print(f"  {key:<5} MISSING  {spec['repo']}  (~{spec['gb']} GB) — {spec['why']}")

    kokoro = [f for f in KOKORO_FILES if (MODELS / "kokoro" / f).exists()]
    if len(kokoro) == len(KOKORO_FILES):
        print(f"  {'tts':<5} present  kokoro  (M5)")
    else:
        print(f"  {'tts':<5} MISSING  kokoro — run --restore-kokoro (M5)")
    return missing


def restore_kokoro() -> int:
    """Copy Kokoro back out of the backup rather than redownloading 350 MB."""
    target = MODELS / "kokoro"
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in KOKORO_FILES:
        source = KOKORO_BACKUP / name
        destination = target / name
        if destination.exists():
            print(f"  {name}: already there")
            continue
        if not source.exists():
            print(f"  {name}: NOT in the backup at {source}")
            continue
        shutil.copy2(source, destination)
        copied += 1
        print(f"  {name}: copied ({destination.stat().st_size / 1024**2:.0f} MB)")
    return copied


def fetch(key: str, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    spec = TARGETS[key]
    path = target_path(key)

    present = list(path.glob("*.safetensors")) + list(path.glob("*.onnx"))
    if not force and present:
        print(f"{key}: already in {path}, skipping (--force to redownload)")
        return path

    available = free_gb(MODELS)
    if available < spec["gb"] * 1.15:
        raise SystemExit(
            f"{key}: needs about {spec['gb']} GB but only {available:.1f} GB is free on "
            f"the volume holding {MODELS}. Free some space, or move models/ elsewhere."
        )

    print(f"{key}: downloading {spec['repo']} (~{spec['gb']} GB) into {path}")
    snapshot_download(
        repo_id=spec["repo"],
        local_dir=str(path),
        allow_patterns=spec["allow"],
        max_workers=4,
    )
    print(f"{key}: done — {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(TARGETS), help="fetch just one target")
    parser.add_argument("--check", action="store_true", help="report what is present, download nothing")
    parser.add_argument("--force", action="store_true", help="redownload even if present")
    parser.add_argument(
        "--restore-kokoro", action="store_true", help="copy Kokoro out of the backup"
    )
    args = parser.parse_args(argv)

    if args.check:
        return 0 if check() == 0 else 1

    if args.restore_kokoro:
        restore_kokoro()
        return 0

    MODELS.mkdir(parents=True, exist_ok=True)
    for key in [args.only] if args.only else list(TARGETS):
        fetch(key, force=args.force)

    print("\nNow verify the model loads and matches PLAN.md's shape claims:")
    print("    python -m src.llm.engine --info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
