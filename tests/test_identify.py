"""Tests for Section 4: mechanistic identification of belief-states.

These tests exploit the mock backend's *known* ground-truth: the identifier
should recover the embedded belief direction at the correct layer, the three
necessary conditions should pass on future-relevant contrasts, and null
(surface-only) contrasts should fail them.
"""

from __future__ import annotations

import numpy as np

from beliefstate.identify import BeliefStateIdentifier, _abs_cosine
from beliefstate.model import MockAgent


def test_identifies_ground_truth_direction_and_layer(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    result = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)

    for horizon in ("near_term", "medium_term"):
        # Peak layer matches the mock's embedded peak (allow +-1 layer slack).
        assert abs(result.peak_layer[horizon] - agent.peak_layer(horizon)) <= 1
        # Recovered direction aligns with the ground-truth belief direction.
        align = _abs_cosine(result.directions[horizon],
                            agent.true_direction(horizon))
        assert align > 0.8


def test_three_necessary_conditions_pass(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    result = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    assert result.conditions["contrastive_sensitivity"]
    assert result.conditions["temporal_persistence"]
    assert result.conditions["horizon_differentiation"]
    assert result.conditions["cross_variant_robustness"]
    assert result.passed


def test_null_sensitivity_below_belief(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    result = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    # Null contrasts must be far weaker than the belief signal.
    assert result.null_sensitivity < cfg.identify.max_null_sensitivity


def test_persistence_curve_higher_for_belief_than_null(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    result = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    assert result.persistence_belief.mean() > result.persistence_null.mean()
    # Belief curve stays elevated across all reasoning steps.
    assert result.persistence_belief[-1] > 0.5


def test_horizon_directions_are_partially_separated(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    result = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    near = result.directions["near_term"]
    med = result.directions["medium_term"]
    # Partial (not perfect) separation: neither identical nor orthogonal.
    cos = _abs_cosine(near, med)
    assert 0.1 < cos < 0.95
