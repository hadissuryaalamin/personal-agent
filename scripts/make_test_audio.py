"""Synthesise spoken commands for testing the voice loop.

M4's exit criterion is that speaking a command produces the correct database
write. Verifying that by hand needs a person and a microphone every time, which
is no basis for a regression test -- so this generates the audio with the
Windows speech synthesiser that ships with the OS. No extra dependency, no
network, and the result is a real 16 kHz mono waveform going through the real
VAD and the real ASR.

    python scripts\\make_test_audio.py            write the standard set
    python scripts\\make_test_audio.py --say "..."  one-off

Synthetic speech is not human speech: it is evenly paced, never trails off, and
has no room tone. Passing here means the *pipeline* works, not that the ASR
copes with how anyone actually talks. The human test still has to happen; this
just means it does not have to happen for every change.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "test_audio"

#: One file per command, named after what it should do.
COMMANDS = {
    "add_class": "put comp four zero two zero on thursdays from nine A M to eleven A M",
    "add_assignment": "add the data structures assignment due next friday, about six hours of work",
    "list_assignments": "what have I got due this week",
    "set_progress": "mark the data structures one as sixty percent done",
    "list_schedule": "what is on today",
    "chat": "thanks, that is great",
    "goodbye": "goodbye",
}

_PS_TEMPLATE = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, `
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, `
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile('{path}', $fmt)
$synth.Rate = {rate}
$synth.Speak('{text}')
$synth.Dispose()
"""


def synthesise(text: str, path: Path, rate: int = 0) -> Path:
    """Windows SAPI to a 16 kHz mono 16-bit wav -- what Silero and Parakeet want."""
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = text.replace("'", "''")
    script = _PS_TEMPLATE.format(path=str(path), text=escaped, rate=rate)
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not path.exists():
        raise SystemExit(f"synthesis failed for {text!r}:\n{result.stderr}")
    return path


def pad_with_silence(path: Path, lead: float = 0.3, tail: float = 0.6) -> None:
    """Give the VAD something to detect an endpoint against.

    Silero closes a segment after `min_silence` of quiet. A file that ends the
    instant the speaker does relies on flush() to emit the last utterance,
    which is not the path the microphone takes -- so the test audio should look
    like the microphone's.
    """
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from src.audio import capture

    samples, rate = capture.read_wav(path)
    padded = np.concatenate(
        [
            np.zeros(int(lead * rate), dtype=np.float32),
            samples,
            np.zeros(int(tail * rate), dtype=np.float32),
        ]
    )
    capture.write_wav(path, padded, rate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="synthesise test commands")
    parser.add_argument("--say", default=None, help="synthesise one phrase")
    parser.add_argument("--out", default=None, help="output path for --say")
    parser.add_argument("--rate", type=int, default=0, help="SAPI rate, -10..10")
    args = parser.parse_args(argv)

    if args.say:
        path = Path(args.out) if args.out else OUT_DIR / "once.wav"
        synthesise(args.say, path, args.rate)
        pad_with_silence(path)
        print(f"wrote {path}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in COMMANDS.items():
        path = OUT_DIR / f"{name}.wav"
        synthesise(text, path, args.rate)
        pad_with_silence(path)
        print(f"  {name:<18} {path.name}  “{text}”")

    # One file with several commands, so the VAD has to split a session up.
    conversation = OUT_DIR / "conversation.wav"
    _join(
        [OUT_DIR / f"{n}.wav" for n in
         ("add_class", "add_assignment", "list_assignments", "set_progress")],
        conversation,
    )
    print(f"  {'conversation':<18} {conversation.name}  (four turns)")
    return 0


def _join(paths: list[Path], out: Path, gap: float = 1.0) -> None:
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from src.audio import capture

    pieces = []
    rate = capture.SAMPLE_RATE
    for path in paths:
        samples, rate = capture.read_wav(path)
        pieces.append(samples)
        pieces.append(np.zeros(int(gap * rate), dtype=np.float32))
    capture.write_wav(out, np.concatenate(pieces), rate)


if __name__ == "__main__":
    sys.exit(main())
