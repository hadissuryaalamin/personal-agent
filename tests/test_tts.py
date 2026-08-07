"""Speech out (M5), without a speaker.

Sentence splitting and the playback queue are the parts with logic in them, and
neither needs an audio device. Whether Kokoro sounds right is in
test_hardware.py; whether it is fast enough is in docs/eval.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.playback import NullSpeaker
from src.tts.kokoro import (
    FIRST_MAX_CHARS,
    FIRST_MIN_CHARS,
    choose_provider,
    split_sentences,
)


# -- splitting a reply into speakable pieces -------------------------------


def test_a_single_sentence_is_one_piece():
    assert split_sentences("Added, due Friday the fourteenth.") == [
        "Added, due Friday the fourteenth."
    ]


def test_sentences_split_on_the_full_stop():
    assert split_sentences("Added, due Friday. Nothing else on.") == [
        "Added, due Friday.",
        "Nothing else on.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "COMP4020 at 9 a.m. then COMP3500.",
        "It is due at 5 p.m. on Friday.",
        "Ask Dr. Smith about it.",
        "Bring a laptop, a charger, etc. and your notes.",
    ],
)
def test_abbreviations_do_not_split_a_sentence(text):
    """"9 a." spoken as a fragment is worse than a slightly long sentence."""
    assert len(split_sentences(text)) == 1, split_sentences(text)


def test_questions_and_exclamations_split_too():
    assert len(split_sentences("Which one? The essay! Or the report.")) == 3


def test_empty_text_produces_nothing():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences(None) == []


def test_a_very_long_sentence_is_broken_up():
    """The first sound must not wait for a 300-character sentence."""
    long = "Three things due this week — " + ", ".join(
        f"the {n} assignment" for n in ["first", "second", "third", "fourth", "fifth", "sixth"]
    ) + "."
    pieces = split_sentences(long, max_chars=80)
    assert len(pieces) > 1
    assert all(len(p) <= 120 for p in pieces), [len(p) for p in pieces]


def test_breaking_a_long_sentence_keeps_all_the_words():
    long = "One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve."
    joined = " ".join(split_sentences(long, max_chars=30))
    for word in ("one", "twelve", "seven"):
        assert word in joined.lower()


def test_the_replies_format_produces_are_one_or_two_pieces():
    """src/format.py caps replies at two sentences, so TTS should see one or two."""
    from datetime import date

    from src import format

    replies = [
        format.reply_for({"tool": "get_now", "time": "09:00", "weekday": "Thursday"},
                         date(2026, 8, 6)),
        format.reply_for({"created": "assignment", "due_date": "2026-08-14",
                          "due": "2026-08-14T23:59:00", "explicit_time": False,
                          "tool": "add_assignment"}, date(2026, 8, 6)),
    ]
    for reply in replies:
        assert 1 <= len(split_sentences(reply)) <= 3, reply


# -- the opening piece, which is the whole latency budget -------------------


def test_a_short_reply_is_still_one_piece():
    """The common case must not be chopped up for a budget it already meets."""
    for reply in ("Added, due Friday the fourteenth.", "Nothing on tomorrow."):
        assert split_sentences(reply, first_max_chars=FIRST_MAX_CHARS) == [reply]


def test_a_long_opening_sentence_is_broken_at_the_dash():
    pieces = split_sentences(
        "Three things due this week — the closest is the data structures assignment, Friday.",
        first_max_chars=FIRST_MAX_CHARS,
    )
    assert pieces[0] == "Three things due this week"
    assert pieces[1].startswith("the closest is")


def test_the_opening_break_falls_back_to_a_comma():
    pieces = split_sentences(
        "Marked sixty percent done, about two and a half hours left.",
        first_max_chars=FIRST_MAX_CHARS,
    )
    assert pieces[0] == "Marked sixty percent done"


def test_the_opening_piece_is_never_a_fragment():
    """"Added" alone, then a pause, sounds like the agent lost its thread."""
    pieces = split_sentences(
        "Added, the data structures assignment is due on Friday the fourteenth.",
        first_max_chars=FIRST_MAX_CHARS,
    )
    assert len(pieces[0]) >= FIRST_MIN_CHARS, pieces


def test_a_sentence_with_no_natural_pause_is_left_whole():
    """A cut mid-clause is worse than waiting: no bare-space breaks up front."""
    text = "The intelligent systems lecture has been moved to the Copland building today"
    assert split_sentences(text, first_max_chars=FIRST_MAX_CHARS) == [text]


def test_breaking_the_opening_keeps_every_word():
    text = "Two classes today — intelligent systems at ten, then the studio at two."
    pieces = split_sentences(text, first_max_chars=FIRST_MAX_CHARS)
    rejoined = " ".join(pieces).lower()
    for word in ("two", "classes", "intelligent", "studio"):
        assert word in rejoined


def test_the_opening_cap_is_off_by_default():
    """split_sentences keeps its old behaviour; only stream() opts in."""
    text = "Three things due this week — the closest is the data structures assignment, Friday."
    assert split_sentences(text) == [text]


# -- picking an execution provider -----------------------------------------


def test_auto_prefers_cuda_when_it_is_there():
    assert choose_provider("auto", ["CPUExecutionProvider", "CUDAExecutionProvider"]) == (
        "CUDAExecutionProvider"
    )


def test_auto_falls_back_to_cpu():
    assert choose_provider("auto", ["CPUExecutionProvider"]) == "CPUExecutionProvider"


def test_auto_copes_with_an_onnxruntime_that_offers_nothing():
    assert choose_provider("auto", []) == "CPUExecutionProvider"


def test_an_explicit_provider_that_is_missing_is_an_error():
    """Pinning CUDA and silently getting the CPU is the bug this prevents."""
    with pytest.raises(ValueError, match="CUDAExecutionProvider"):
        choose_provider("CUDAExecutionProvider", ["CPUExecutionProvider"])


def test_an_explicit_provider_that_is_present_is_honoured():
    available = ["CPUExecutionProvider", "CUDAExecutionProvider"]
    assert choose_provider("CPUExecutionProvider", available) == "CPUExecutionProvider"


# -- the speaker queue -----------------------------------------------------


def test_null_speaker_collects_instead_of_playing():
    speaker = NullSpeaker(sample_rate=24000)
    speaker.say(np.zeros(24000, dtype=np.float32))
    speaker.say(np.zeros(12000, dtype=np.float32))
    assert speaker.seconds == pytest.approx(1.5)


def test_null_speaker_counts_interruptions():
    speaker = NullSpeaker()
    speaker.interrupt()
    speaker.interrupt()
    assert speaker.interrupted == 2


def test_null_speaker_satisfies_the_same_interface():
    from src.audio.playback import Speaker

    for name in ("say", "interrupt", "wait", "close", "start", "playing"):
        assert hasattr(NullSpeaker(), name), name
        assert hasattr(Speaker, name), name


def test_speaker_is_not_opened_until_used():
    """Constructing one must not need an audio device."""
    from src.audio.playback import Speaker

    assert Speaker()._thread is None


def test_interrupting_an_idle_speaker_is_harmless():
    from src.audio.playback import Speaker

    assert Speaker().interrupt() == 0


# -- wiring ---------------------------------------------------------------


def test_kokoro_says_where_to_get_the_weights(tmp_path):
    from src.tts.kokoro import Kokoro

    with pytest.raises(FileNotFoundError, match="restore-kokoro"):
        Kokoro(model_path=tmp_path / "nope.onnx", voices_path=tmp_path / "v.bin").load()


def test_the_voice_is_english():
    """D4: English speech in and out."""
    from src.tts import kokoro

    assert kokoro.DEFAULT_LANG.startswith("en")
    assert kokoro.DEFAULT_VOICE.startswith(("bf_", "bm_", "af_", "am_"))


def test_a_session_without_tts_does_not_speak(conn, cfg):
    from src.session import VoiceSession

    session = VoiceSession(conn, cfg, tts=None, speaker=NullSpeaker())
    assert session.speak("Added, due Friday.") is None


def test_a_session_without_a_speaker_does_not_speak(conn, cfg):
    from src.session import VoiceSession

    session = VoiceSession(conn, cfg, tts=object(), speaker=None)
    assert session.speak("Added, due Friday.") is None


def test_an_empty_reply_is_not_spoken(conn, cfg):
    from src.session import VoiceSession

    session = VoiceSession(conn, cfg, tts=object(), speaker=NullSpeaker())
    assert session.speak("   ") is None
