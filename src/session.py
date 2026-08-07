"""The voice turn loop (M4).

    python -m src.session                     listen on the default microphone
    python -m src.session --wav command.wav   run the same pipeline on a file
    python -m src.session --devices           list input devices and exit

    mic ─► VAD ─► Parakeet ─► the same Repl the text mode uses ─► SQLite

Everything below the transcript is shared with `src.cli`: the same gate, the
same tools, the same confirmations, the same `turn_log` row. Voice adds two
things and no more -- turning sound into a transcript, and knowing when the
speaker has stopped. That is deliberate. Behaviour that only exists in the
voice path could only be tested by talking to it.

`--wav` exists because the exit criterion for M4 is "speaking a command
produces the correct database write", and a wav file is a command someone
spoke. It is also the only way to get a regression test out of a microphone.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.audio import capture, playback
from src.audio.vad import Segmenter
from src.asr.parakeet import Parakeet
from src.cli import Repl, ensure_schema
from src.store.db import connect
from src.turnlog import record_timing

#: Said out loud, these close the session. README: "Press it again, or say
#: goodbye, to close."
GOODBYE = (
    "goodbye", "good bye", "bye", "bye bye", "that is all", "that's all",
    "we are done", "we're done", "that will do", "stop listening",
    "see you later", "goodnight", "good night",
)

#: Below this, a segment is a cough or a chair. Parakeet will happily
#: hallucinate a sentence out of half a second of noise.
MIN_SEGMENT_SECONDS = 0.4


def is_goodbye(text: str) -> bool:
    cleaned = " ".join(text.lower().strip(" .,!?").split())
    return cleaned in GOODBYE


class VoiceSession:
    """One open session: many turns, one microphone."""

    def __init__(
        self,
        conn,
        cfg: config.Config | None = None,
        repl: Repl | None = None,
        asr: Parakeet | None = None,
        segmenter: Segmenter | None = None,
        tts=None,
        speaker=None,
        on_reply=None,
        barge_in: bool = True,
    ) -> None:
        self.cfg = cfg or config.load()
        self.conn = conn
        self.session_id = uuid.uuid4().hex[:12]
        self.repl = repl or Repl(conn, self.cfg, session_id=self.session_id)
        self.asr = asr or Parakeet()
        self.segmenter = segmenter or Segmenter()
        self.tts = tts
        self.speaker = speaker
        self.barge_in = barge_in
        self.on_reply = on_reply or (lambda text, reply: None)
        self.turns = 0
        self.interruptions = 0

    def load(self) -> "VoiceSession":
        """Warm everything before the first word, not during it.

        The first Kokoro call costs about four seconds of graph setup and the
        first Parakeet call is not much better. Paying that at startup rather
        than in the middle of the user's first sentence is most of the
        difference between "slow" and "broken".
        """
        self.asr.load()
        self.segmenter.load()
        if self.tts is not None:
            self.tts.load()
            self.tts.synthesise("ready")
        return self

    # -- speaking ---------------------------------------------------------

    def speak(self, reply: str) -> int | None:
        """Stream the reply to the speaker. Returns ms to the first audio."""
        if self.tts is None or self.speaker is None or not reply.strip():
            return None

        first_ms = None
        started = time.perf_counter()
        for chunk in self.tts.stream(reply):
            if chunk.first:
                first_ms = int((time.perf_counter() - started) * 1000)
            self.speaker.say(chunk.samples)
        return first_ms

    # -- one utterance ----------------------------------------------------

    def handle_segment(self, segment) -> tuple[str, str] | None:
        """Transcribe one detected utterance and run it as a turn.

        Returns (transcript, reply), or None if the segment was too short to
        be speech. A segment that transcribes to nothing still gets a turn_log
        row -- invariant #3 is explicit that empty transcripts are logged,
        because "the ASR heard nothing" is the single most useful line in the
        log when someone says the agent has gone deaf.
        """
        if segment.seconds < MIN_SEGMENT_SECONDS:
            return None

        transcript = self.asr.transcribe(segment.samples, segment.sample_rate)
        reply = self.repl.handle(
            transcript.text, spoken=True, ms_asr=transcript.ms
        )
        self.turns += 1

        ms_tts = self.speak(reply)
        if ms_tts is not None and self.repl.last_turn_id is not None:
            record_timing(self.conn, self.repl.last_turn_id, ms_tts=ms_tts)

        self.on_reply(transcript.text, reply)
        return transcript.text, reply

    # -- whole sessions ---------------------------------------------------

    def run_wav(self, path: Path | str) -> list[tuple[str, str]]:
        """Run the pipeline over a recording, in real-time-sized blocks."""
        self.load()
        samples, rate = capture.read_wav(path)
        if rate != capture.SAMPLE_RATE:
            raise ValueError(
                f"{path} is {rate} Hz; the VAD and Parakeet both want "
                f"{capture.SAMPLE_RATE} Hz"
            )

        out = []
        for start in range(0, len(samples), capture.BLOCK_SAMPLES):
            for segment in self.segmenter.push(samples[start : start + capture.BLOCK_SAMPLES]):
                handled = self.handle_segment(segment)
                if handled:
                    out.append(handled)
        for segment in self.segmenter.flush():
            handled = self.handle_segment(segment)
            if handled:
                out.append(handled)
        return out

    def run_microphone(self, device=None, max_seconds: float | None = None) -> None:
        self.load()
        started = time.perf_counter()

        with capture.Microphone(device=device) as mic:
            print("listening — say goodbye, or press Ctrl+C, to close.\n", flush=True)
            for block in mic.blocks():
                segments = self.segmenter.push(block)

                # Barge-in: the microphone stays open while the agent talks, so
                # starting to speak cuts it off mid-word.
                if (
                    self.barge_in
                    and self.speaker is not None
                    and self.speaker.playing
                    and (segments or self.segmenter.speaking)
                ):
                    self.speaker.interrupt()
                    self.interruptions += 1

                for segment in segments:
                    handled = self.handle_segment(segment)
                    if handled and is_goodbye(handled[0]):
                        return
                if max_seconds and time.perf_counter() - started > max_seconds:
                    break

            for segment in self.segmenter.flush():
                self.handle_segment(segment)

        if mic.dropped:
            print(f"note: {mic.dropped} audio blocks were dropped", flush=True)


def _print_turn(transcript: str, reply: str) -> None:
    print(f"  heard: {transcript!r}" if transcript else "  heard: (nothing)", flush=True)
    print(f"  {reply}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="personal-agent voice loop")
    parser.add_argument("--wav", default=None, help="run on a recording instead of the mic")
    parser.add_argument("--devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--device", default=None, help="input device index or name")
    parser.add_argument("--output-device", default=None, help="output device index or name")
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    parser.add_argument("--silent", action="store_true", help="do not speak the replies")
    parser.add_argument(
        "--no-barge-in", action="store_true",
        help="do not let speech interrupt playback (use on laptop speakers)",
    )
    parser.add_argument("--save-audio", default=None, help="write spoken replies to a wav")
    args = parser.parse_args(argv)

    if args.devices:
        inputs = capture.list_input_devices()
        outputs = playback.list_output_devices()
        print("input:")
        for device in inputs or []:
            print(f"  {device.index:>2}  {device.name}{' (default)' if device.default else ''}")
        print("\noutput:")
        for device in outputs or []:
            print(f"  {device.index:>2}  {device.name}{' (default)' if device.default else ''}")
        return 0 if inputs and outputs else 1

    cfg = config.load()
    conn = connect(cfg.db_path)
    ensure_schema(conn)

    now = datetime.now(timezone.utc).astimezone(cfg.tz)
    print(f"personal-agent — voice. {now:%A %d %B %Y, %H:%M} {cfg.tz_name}")
    print(f"gate: {cfg.gate}")

    tts = speaker = None
    if not args.silent:
        from src.tts.kokoro import Kokoro

        tts = Kokoro()
        if args.wav or args.save_audio:
            speaker = playback.NullSpeaker(sample_rate=tts_rate(tts))
        else:
            device = _as_device(args.output_device)
            speaker = playback.Speaker(sample_rate=tts_rate(tts), device=device)
            # A silent agent with a clean log is nearly always the speaker
            # moving, not the code breaking. Say which one it is, up front.
            print(f"speaking through: {playback.describe_output(device)}")

    session = VoiceSession(
        conn, cfg, tts=tts, speaker=speaker,
        on_reply=_print_turn, barge_in=not args.no_barge_in,
    )

    try:
        if args.wav:
            session.run_wav(args.wav)
        else:
            session.run_microphone(
                device=_as_device(args.device), max_seconds=args.seconds
            )
    except KeyboardInterrupt:
        print()
    finally:
        if speaker is not None:
            if args.save_audio and isinstance(speaker, playback.NullSpeaker):
                _save(speaker, args.save_audio)
            speaker.close()
        print(f"{session.turns} turns this session.", end="")
        print(f" {session.interruptions} interrupted." if session.interruptions else "")
        conn.close()
    return 0


def tts_rate(tts) -> int:
    from src.tts import kokoro

    return kokoro.SAMPLE_RATE


def _as_device(value):
    if value is None:
        return None
    return int(value) if str(value).isdigit() else value


def _save(speaker, path) -> None:
    import numpy as np

    if not speaker.chunks:
        print("no audio to save")
        return
    capture.write_wav(path, np.concatenate(speaker.chunks), speaker.sample_rate)
    print(f"wrote {path} ({speaker.seconds:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
