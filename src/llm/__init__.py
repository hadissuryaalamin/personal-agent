"""Model engine, prompts, the tool gate, and the turn planner.

Import submodules directly (``from src.llm import agent``). Nothing is
re-exported here, so importing the package does not drag in torch -- text mode
must stay usable without the weights on disk.
"""
