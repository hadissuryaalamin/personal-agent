"""Tests that need the real weights on disk and a GPU.

Excluded from the default run (invariant #7). To run them:

    pytest -m hardware

These are the checks that cannot be faked: that the model on disk is the shape
PLAN.md assumes, that one prefill really can drive two passes, and that the
prompted gate separates a request from a remark at all -- which is the premise
the probe is measured against at M3.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.llm import prompts
from src.llm.engine import Engine
from src.llm.gate import PromptedGate
from src.tools import registry

pytestmark = pytest.mark.hardware

TZ = ZoneInfo("Australia/Sydney")
NOW = datetime(2026, 8, 6, 10, 0, tzinfo=TZ)


@pytest.fixture(scope="module")
def engine():
    return Engine().load()


@pytest.fixture(scope="module")
def gate():
    return PromptedGate()


def prefill_for(engine, text: str):
    messages = prompts.build_messages(NOW, "Australia/Sydney", text)
    return engine.prefill(engine.prefix_text(messages))


# -- the model is what the docs assume -------------------------------------


def test_model_shape_matches_the_plan(engine):
    """PLAN.md section 4 says 36 layers, hidden 2560 -- check, do not trust."""
    assert engine.info["layers"] == 36
    assert engine.info["hidden_size"] == 2560


def test_hidden_states_cover_every_layer(engine):
    prefill = prefill_for(engine, "what is due next friday")
    assert prefill.hidden.shape == (engine.info["layers"] + 1, engine.info["hidden_size"])
    assert prefill.hidden.any(), "an all-zero hidden state means the capture is wrong"


def test_layer_accessor_matches_the_row(engine):
    prefill = prefill_for(engine, "hello")
    assert (prefill.layer(18) == prefill.hidden[18]).all()


# -- one prefill, two passes -----------------------------------------------


def test_the_cache_survives_a_continuation(engine):
    """The gate and the tool pass both run from the same prefilled prefix."""
    prefill = prefill_for(engine, "what is on tomorrow")
    before = prefill.cache.get_seq_length()

    engine.continue_from(prefill, "<|im_start|>assistant\n", max_new_tokens=4)
    assert prefill.cache.get_seq_length() == before

    engine.continue_from(prefill, "<|im_start|>assistant\n", max_new_tokens=4)
    assert prefill.cache.get_seq_length() == before


def test_two_passes_from_one_prefill_agree(engine):
    """Same cache, same suffix, same greedy output -- no cache contamination."""
    prefill = prefill_for(engine, "what is on tomorrow")
    first, _ = engine.continue_from(prefill, "<|im_start|>assistant\n", max_new_tokens=8)
    second, _ = engine.continue_from(prefill, "<|im_start|>assistant\n", max_new_tokens=8)
    assert first == second


# -- the gate, against the numbers docs/eval.md actually reports -----------
#
# These thresholds are floors below the measured result, not targets. The point
# is to catch a regression in the prompt or the scoring, not to assert that the
# prompted baseline is good -- it is not, and that is the finding. Run
# `python scripts\eval_gate.py` for the current numbers.

#: Measured 77.5% on data/gate_seed.jsonl at the time of writing.
MIN_ACCURACY = 0.70
#: Measured 10%. A false skip is merely unhelpful; a false call writes data.
MAX_FALSE_SKIP = 0.25


@pytest.fixture(scope="module")
def seed_results(engine, gate):
    from scripts.eval_gate import load_dataset, measure, summarise

    return summarise(measure(engine, gate, load_dataset()))


def test_gate_accuracy_has_not_regressed(seed_results):
    assert seed_results["accuracy"] >= MIN_ACCURACY, (
        f"{seed_results['accuracy']:.1%} on the seed set; "
        f"misses: {[r['text'] for r in seed_results['false_calls'] + seed_results['false_skips']]}"
    )


def test_the_gate_still_catches_most_real_requests(seed_results):
    assert seed_results["false_skip_rate"] <= MAX_FALSE_SKIP


def test_unambiguous_requests_and_remarks_land_on_the_right_side(engine, gate):
    """The cases the baseline does get right, as a canary."""
    request = gate.decide(engine, prefill_for(engine, "what is on tomorrow"))
    remark = gate.decide(engine, prefill_for(engine, "hello"))
    assert request.score > 0.5 > remark.score


# -- the second pass produces a callable tool ------------------------------


@pytest.fixture(scope="module")
def agent(engine):
    from src import config
    from src.llm.agent import Agent

    return Agent(engine, config.load())


@pytest.mark.parametrize(
    "text,expected",
    [
        ("what is on tomorrow", "list_schedule"),
        ("what have I got due this week", "list_assignments"),
        ("add the essay assignment due next friday", "add_assignment"),
    ],
)
def test_the_second_pass_names_a_real_tool(agent, text, expected):
    """Forced down the tool branch: the gate is measured separately above."""
    from src.llm.engine import Prefill  # noqa: F401  (documents the return type)

    plan = agent._extract_call(prefill_for(agent.engine, text), _blank_plan())
    assert plan.tool in registry.TOOLS, f"invented tool {plan.tool!r} from {plan.raw!r}"
    assert plan.tool == expected, f"{text!r} -> {plan.tool} {plan.args}"


def test_time_expressions_are_passed_through_not_resolved(agent):
    """Invariant #1: the model must not turn "next friday" into a date."""
    plan = agent._extract_call(
        prefill_for(agent.engine, "add the essay assignment due next friday"),
        _blank_plan(),
    )
    assert plan.tool == "add_assignment"
    due = str(plan.args.get("due", "")).lower()
    assert "friday" in due, f"expected the phrase back, got {due!r}"
    assert "2026" not in due and "-" not in due, f"the model did date arithmetic: {due!r}"


def _blank_plan():
    from src.llm.agent import Plan

    return Plan(label="tool")


# -- the probe, against the real model (M3) --------------------------------


@pytest.fixture(scope="module")
def probe():
    from src.config import ROOT
    from src.llm.gate import ProbeGate

    path = ROOT / "data" / "probe.joblib"
    if not path.exists():
        pytest.skip("no probe — run scripts/train_probe.py")
    return ProbeGate(path).load()


def test_the_probe_matches_the_model_it_was_trained_on(engine, probe):
    """A probe from different weights would produce confident nonsense."""
    assert probe.n_layers == engine.info["layers"]
    assert probe.hidden_size == engine.info["hidden_size"]


def test_the_probe_reads_a_real_prefill(engine, probe):
    decision = probe.decide(engine, prefill_for(engine, "what is on tomorrow"))
    assert decision.source == "probe"
    assert 0.0 <= decision.score <= 1.0


@pytest.mark.parametrize("text", ["what is on tomorrow", "add an essay due friday"])
def test_requests_score_high_through_the_probe(engine, probe, text):
    assert probe.decide(engine, prefill_for(engine, text)).score > 0.5


@pytest.mark.parametrize("text", ["hello", "that is annoying", "what can you do"])
def test_remarks_score_low_through_the_probe(engine, probe, text):
    """The three the prompted gate gets wrong. This is the M3 point."""
    assert probe.decide(engine, prefill_for(engine, text)).score < 0.5


def test_the_probe_is_faster_than_the_prompted_gate(engine, gate, probe):
    """PLAN.md section 4: the probe must beat the baseline on latency too."""
    import time

    prefill = prefill_for(engine, "what have I got due this week")
    probe.decide(engine, prefill)  # warm

    started = time.perf_counter()
    for _ in range(5):
        probe.decide(engine, prefill)
    probe_ms = (time.perf_counter() - started) * 1000 / 5

    started = time.perf_counter()
    for _ in range(5):
        gate.decide(engine, prefill)
    prompted_ms = (time.perf_counter() - started) * 1000 / 5

    assert probe_ms < prompted_ms, f"probe {probe_ms:.1f} ms vs prompted {prompted_ms:.1f} ms"


def test_one_prefill_serves_both_gates(engine, gate, probe):
    """Neither gate may corrupt the cache the other one needs."""
    prefill = prefill_for(engine, "what is on tomorrow")
    before = prefill.cache.get_seq_length()

    gate.decide(engine, prefill)
    probe.decide(engine, prefill)
    gate.decide(engine, prefill)

    assert prefill.cache.get_seq_length() == before


# -- voice in (M4) ---------------------------------------------------------
#
# These need the ASR and VAD weights but no microphone: the audio is
# synthesised by scripts/make_test_audio.py, which uses the speech synthesiser
# built into Windows.


@pytest.fixture(scope="module")
def asr():
    from src.asr.parakeet import Parakeet

    recogniser = Parakeet()
    if not recogniser.model_dir.exists():
        pytest.skip("no ASR model — run scripts/fetch_models.py --only asr")
    return recogniser.load()


@pytest.fixture(scope="module")
def test_audio():
    from src.config import ROOT

    directory = ROOT / "data" / "test_audio"
    if not (directory / "add_assignment.wav").exists():
        pytest.skip("no test audio — run scripts/make_test_audio.py")
    return directory


def test_parakeet_transcribes_its_own_sample(asr):
    from src.audio import capture

    wav = asr.model_dir / "test_wavs" / "0.wav"
    if not wav.exists():
        pytest.skip("no sample wav in the model directory")

    samples, rate = capture.read_wav(wav)
    result = asr.transcribe(samples, rate)
    assert "Phebe" in result.text or "phebe" in result.text.lower()


def test_asr_is_inside_its_latency_budget(asr, test_audio):
    """PLAN.md section 5 gives Parakeet 300 ms."""
    from src.audio import capture

    samples, rate = capture.read_wav(test_audio / "list_assignments.wav")
    asr.transcribe(samples, rate)  # warm

    result = asr.transcribe(samples, rate)
    assert result.ms < 300, f"{result.ms} ms for {result.seconds_of_audio:.1f}s"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("list_assignments", "due this week"),
        ("list_schedule", "on today"),
        ("set_progress", "sixty percent"),
        ("goodbye", "goodbye"),
    ],
)
def test_the_asr_hears_the_command(asr, test_audio, name, expected):
    from src.audio import capture

    samples, rate = capture.read_wav(test_audio / f"{name}.wav")
    heard = asr.transcribe(samples, rate).text.lower()
    assert expected in heard, f"heard {heard!r}"


def test_the_vad_finds_one_utterance_per_file(test_audio):
    from src.audio import capture
    from src.audio.vad import Segmenter

    segmenter = Segmenter()
    if not segmenter.model_path.exists():
        pytest.skip("no VAD model — run scripts/fetch_models.py --only vad")

    samples, _ = capture.read_wav(test_audio / "add_assignment.wav")
    segments = []
    for start in range(0, len(samples), capture.BLOCK_SAMPLES):
        segments += segmenter.push(samples[start : start + capture.BLOCK_SAMPLES])
    segments += segmenter.flush()

    assert len(segments) == 1, f"expected one utterance, got {len(segments)}"
    assert segments[0].seconds > 1.0


def test_the_vad_splits_a_conversation_into_turns(test_audio):
    from src.audio import capture
    from src.audio.vad import Segmenter

    segmenter = Segmenter()
    if not segmenter.model_path.exists():
        pytest.skip("no VAD model")

    samples, _ = capture.read_wav(test_audio / "conversation.wav")
    segments = []
    for start in range(0, len(samples), capture.BLOCK_SAMPLES):
        segments += segmenter.push(samples[start : start + capture.BLOCK_SAMPLES])
    segments += segmenter.flush()

    assert len(segments) == 4, f"expected four turns, got {len(segments)}"


def test_kokoro_speaks_a_reply(tts):
    chunk = tts.synthesise("Added, due Friday the fourteenth.")
    assert chunk.seconds > 0.8, "a seven-word sentence should be about two seconds"
    assert abs(chunk.samples).max() > 0.01, "silence means the voice did not load"


def test_kokoro_streams_a_sentence_at_a_time(tts):
    reply = "Added, due Friday the fourteenth. Nothing else is on today."
    chunks = list(tts.stream(reply))

    assert len(chunks) == 2
    assert chunks[0].first and not chunks[1].first
    # The first chunk is what the user waits for, so it must be the short one.
    assert chunks[0].seconds < sum(c.seconds for c in chunks)


def test_the_first_chunk_is_what_the_budget_is_about(tts):
    """PLAN.md section 5 allows 250 ms to the first audio. Measured, not assumed."""
    tts.synthesise("warm")
    chunks = list(tts.stream("Nothing on today. Three things are due this week."))

    # scripts/bench_tts.py measures 157 ms on CUDA and 504 ms on the CPU. The
    # limits are loose enough not to flake on a busy machine and tight enough
    # to catch the failure that actually happens: onnxruntime advertising a
    # provider it cannot build and quietly running on the CPU instead.
    on_gpu = "CUDA" in (tts.provider_in_use or "")
    limit = 400 if on_gpu else 1200
    assert chunks[0].ms < limit, (
        f"{chunks[0].ms} ms to first audio on {tts.provider_in_use}, over the "
        f"{limit} ms this configuration should manage — see docs/eval.md"
    )


def test_the_tts_does_not_break_torch():
    """Kokoro on CUDA must not stop the model loading afterwards.

    onnxruntime and torch both want a DLL called cudnn64_9.dll, built against
    different CUDA majors, and Windows keeps one per name per process. When the
    TTS moved to the GPU this broke every turn of the voice loop: the model
    failed to load with WinError 127 naming *torch's* file, so nothing pointed
    at the TTS. src/tts/kokoro.py now imports torch first and keeps the CUDA
    libraries off PATH once the session is built.

    Loads in the order the session does, which is the order that failed.
    """
    from src.tts.kokoro import Kokoro

    speech = Kokoro()
    if not speech.model_path.exists():
        pytest.skip("no TTS weights — run scripts/fetch_models.py --restore-kokoro")
    speech.load()
    speech.synthesise("Added, due Friday the fourteenth.")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU")

    # A real cuDNN call, not just the import: this is what raised.
    tensor = torch.randn(1, 8, 16, 16, device="cuda")
    weight = torch.randn(8, 8, 3, 3, device="cuda")
    torch.nn.functional.conv2d(tensor, weight)
    torch.cuda.synchronize()

    # And the TTS still works afterwards, on the provider it reported.
    assert speech.synthesise("Nothing on tomorrow.").seconds > 0.3


def test_the_cuda_libraries_do_not_stay_on_path():
    """PATH is process-wide. Leaving the CUDA 13 runtime on it is the bug."""
    import os

    from src.tts.kokoro import Kokoro, cuda_dll_dirs

    speech = Kokoro()
    if not speech.model_path.exists():
        pytest.skip("no TTS weights — run scripts/fetch_models.py --restore-kokoro")
    speech.load()

    path = os.environ.get("PATH", "")
    for directory in cuda_dll_dirs():
        assert directory not in path, f"{directory} left on PATH after load"


@pytest.fixture(scope="module")
def tts():
    from src.tts.kokoro import Kokoro

    engine = Kokoro()
    if not engine.model_path.exists():
        pytest.skip("no TTS weights — run scripts/fetch_models.py --restore-kokoro")
    return engine.load()


def test_silence_produces_no_turns(test_audio):
    import numpy as np

    from src.audio import capture
    from src.audio.vad import Segmenter

    segmenter = Segmenter()
    if not segmenter.model_path.exists():
        pytest.skip("no VAD model")

    quiet = np.zeros(capture.SAMPLE_RATE * 3, dtype=np.float32)
    segments = []
    for start in range(0, len(quiet), capture.BLOCK_SAMPLES):
        segments += segmenter.push(quiet[start : start + capture.BLOCK_SAMPLES])
    segments += segmenter.flush()

    assert segments == []
