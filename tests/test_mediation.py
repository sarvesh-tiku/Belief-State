"""Tests for causal mediation analysis (mediation.py, Vig et al. 2020).

In the mock backend the belief-state fully carries the treatment's effect on
downstream behaviour, so the decomposition should recover a large indirect
effect (NIE), a near-zero direct effect (NDE), and TE ~= NDE + NIE.
"""

from __future__ import annotations

import numpy as np

from beliefstate.identify import BeliefStateIdentifier
from beliefstate.mediation import MediationAnalysis
from beliefstate.model import MockAgent


def _run(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    belief = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    return MediationAnalysis(cfg, agent, rng=rng).run(corpus, belief, "near_term")


def test_effects_decompose_additively(cfg, corpus, rng):
    med = _run(cfg, corpus, rng)
    # TE ~= NDE + NIE (mediation identity under linear patching).
    assert abs(med.total_effect - (med.natural_direct_effect
                                   + med.natural_indirect_effect)) < 1e-4


def test_belief_state_mediates_effect(cfg, corpus, rng):
    med = _run(cfg, corpus, rng)
    # Belief-state carries most of the effect: high proportion mediated.
    assert med.natural_indirect_effect != 0.0
    assert med.proportion_mediated > 0.5
    assert abs(med.natural_direct_effect) < abs(med.natural_indirect_effect)


def test_summary_keys(cfg, corpus, rng):
    med = _run(cfg, corpus, rng)
    s = med.summary()
    assert set(s) >= {
        "horizon", "total_effect", "natural_direct_effect",
        "natural_indirect_effect", "proportion_mediated", "n_pairs",
    }
    assert s["n_pairs"] > 0
