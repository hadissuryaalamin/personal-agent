"""Microphone capture and turn segmentation.

Importing this package must not require an audio device: `src.cli` and the
tests both import things that live under `src/` without ever opening a stream.
Submodules are imported directly (``from src.audio import vad``).
"""
