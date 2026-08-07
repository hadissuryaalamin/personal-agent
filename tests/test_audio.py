"""Audio plumbing that does not need a microphone.

Invariant #7: the default run has no audio device. Everything here works on
arrays and files. What needs real hardware -- opening a stream, and whether
Silero and Parakeet agree with each other -- is in test_hardware.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio import capture
from src.audio.vad import Segment, Segmenter


def tone(seconds=1.0, rate=16000, freq=220.0, amplitude=0.3):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -- wav round trip --------------------------------------------------------


def test_wav_round_trip(tmp_path):
    original = tone(0.5)
    path = tmp_path / "a.wav"
    capture.write_wav(path, original)

    loaded, rate = capture.read_wav(path)
    assert rate == capture.SAMPLE_RATE
    assert len(loaded) == len(original)
    # 16-bit quantisation, so close rather than equal.
    assert np.abs(loaded - original).max() < 1e-3


def test_wav_clips_rather_than_wrapping(tmp_path):
    """A sample above 1.0 must not wrap round to a loud negative."""
    path = tmp_path / "loud.wav"
    capture.write_wav(path, np.array([2.0, -2.0, 0.0], dtype=np.float32))
    loaded, _ = capture.read_wav(path)
    assert loaded[0] > 0.99 and loaded[1] < -0.99


def test_stereo_is_mixed_down(tmp_path):
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(200, dtype=np.int16).tobytes())

    samples, _ = capture.read_wav(path)
    assert samples.ndim == 1
    assert len(samples) == 100


def test_non_16_bit_audio_is_rejected(tmp_path):
    import wave

    path = tmp_path / "eight.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(16000)
        handle.writeframes(b"\x00" * 100)

    with pytest.raises(ValueError, match="16-bit"):
        capture.read_wav(path)


def test_a_block_is_one_silero_window():
    """512 samples at 16 kHz is Silero's window, so the VAD never buffers a partial one."""
    assert capture.BLOCK_SAMPLES == 512
    assert capture.SAMPLE_RATE == 16000


# -- segments --------------------------------------------------------------


def test_segment_reports_its_length():
    segment = Segment(samples=np.zeros(8000, dtype=np.float32), start_sample=16000)
    assert segment.seconds == pytest.approx(0.5)
    assert segment.start_seconds == pytest.approx(1.0)


def test_segmenter_defaults_do_not_cut_people_off():
    """Measured: below ~0.5 s the VAD splits sentences at their commas."""
    from src.audio import vad

    assert 0.5 <= vad.MIN_SILENCE <= 1.0
    assert vad.MIN_SPEECH >= 0.2, "shorter than this and coughs become turns"
    assert vad.MAX_SPEECH <= 30, "a stuck stream must not buffer forever"


def test_segmenter_says_where_to_get_the_model(tmp_path):
    segmenter = Segmenter(model_path=tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError, match="fetch_models"):
        segmenter.load()


def test_segmenter_is_not_loaded_until_used(tmp_path):
    """Constructing one must not need the model or an audio device."""
    assert Segmenter(model_path=tmp_path / "nope.onnx")._vad is None


# -- the ASR wrapper -------------------------------------------------------


def test_parakeet_says_where_to_get_the_model(tmp_path):
    from src.asr.parakeet import Parakeet

    with pytest.raises(FileNotFoundError, match="fetch_models"):
        Parakeet(model_dir=tmp_path / "nope").load()


def test_transcript_reports_real_time_factor():
    from src.asr.parakeet import Transcript

    t = Transcript(text="hello", ms=300, seconds_of_audio=3.0)
    assert t.real_time_factor == pytest.approx(0.1)
    assert not t.empty


def test_an_empty_transcript_is_flagged_not_hidden():
    from src.asr.parakeet import Transcript

    assert Transcript(text="   ", ms=5, seconds_of_audio=1.0).empty


def test_transcribing_nothing_returns_nothing(tmp_path):
    """Zero samples must not reach the recogniser or raise."""
    from src.asr.parakeet import Parakeet

    asr = Parakeet(model_dir=tmp_path / "nope")
    result = asr.transcribe(np.array([], dtype=np.float32))
    assert result.empty and result.ms == 0


def test_confidence_is_none_rather_than_invented():
    from src.asr.parakeet import Transcript

    assert Transcript(text="x", ms=1, seconds_of_audio=1.0).confidence is None
