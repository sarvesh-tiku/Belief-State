"""Tests for the non-stationarity stress test (stress.py, Section 2.4).

The regime-shift condition should produce systematically larger belief-state
drift than the stable condition, and the representation-level instability onset
should precede the operationalized failure step.
"""

from __future__ import annotations

import numpy as np

from beliefstate.identify import BeliefStateIdentifier
from beliefstate.model import MockAgent
from beliefstate.stress import StressTest


def _run(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    belief = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    stress = StressTest(cfg, agent, rng=rng).run(corpus, belief, "near_term")
    return stress


def test_shift_drift_exceeds_stable(cfg, corpus, rng):
    stress = _run(cfg, corpus, rng)
    assert stress.drift_ratio > 1.5
    assert stress.shift_drift.mean() > stress.stable_drift.mean()


def test_instability_onset_precedes_failure(cfg, corpus, rng):
    stress = _run(cfg, corpus, rng)
    assert stress.instability_onset_step is not None
    assert stress.failure_step is not None
    assert stress.instability_onset_step < stress.failure_step
    assert stress.onset_precedes_failure


def test_drift_curves_same_length(cfg, corpus, rng):
    stress = _run(cfg, corpus, rng)
    assert stress.stable_drift.shape == stress.shift_drift.shape
    assert stress.n_instances > 0


def test_summary_keys(cfg, corpus, rng):
    stress = _run(cfg, corpus, rng)
    s = stress.summary()
    assert s["onset_precedes_failure"] is True
    assert s["drift_ratio_shift_over_stable"] > 1.0
