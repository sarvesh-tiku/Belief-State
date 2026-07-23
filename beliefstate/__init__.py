"""BELIEF-STATE: Interpreting Temporal Belief Dynamics in Agentic Financial Systems.

This package implements the full pipeline described in the paper:

    * ``data``          -- Section 3.1, temporally aligned earnings-call/market tuples.
    * ``tasks``         -- Sections 3.2/3.3, horizon-aware tasks, prompts, contrasts.
    * ``model``         -- the agent wrapper: activation capture + steering hooks.
    * ``identify``      -- Section 4, mechanistic identification of belief-states.
    * ``intervene``     -- Section 5, causal activation-level interventions.
    * ``figures``       -- reproduction of Figures 1 and 2.

The design goal is that ``python -m scripts.run_all`` reproduces every empirical
claim and figure end-to-end, using either a real Hugging Face transformer
(``model.backend: hf``) or a deterministic mock transformer (``model.backend:
mock``) whose activations carry a known ground-truth belief-state.
"""

from .config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "1.0.0"
