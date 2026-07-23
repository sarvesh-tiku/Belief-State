"""Tests for the cross-validated linear probes (probes.py).

The probe should (a) decode the future-relevant regime label well above chance,
(b) localize to roughly the same layer as the difference-of-means peak, and
(c) learn a weight vector aligned with the difference-of-means direction --
independent, convergent evidence for the belief-state.
"""

from __future__ import annotations

import numpy as np

from beliefstate.identify import BeliefStateIdentifier
from beliefstate.model import MockAgent
from beliefstate.probes import BeliefProbe


def test_probe_decodes_above_chance(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    pr = BeliefProbe(cfg, agent, rng=rng).run(corpus, "near_term")
    assert pr.best_accuracy > pr.chance + 0.2
    assert pr.n_samples > 0
    # Accuracy profile has one entry per captured layer (incl. embeddings).
    assert pr.layer_accuracy.shape[0] >= agent.num_layers


def test_probe_localizes_near_diff_of_means_peak(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    belief = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    pr = BeliefProbe(cfg, agent, rng=rng).run(
        corpus, "near_term", belief.directions["near_term"]
    )
    # Two independent methods should localize to nearby layers.
    assert abs(pr.best_layer - belief.peak_layer["near_term"]) <= 3


def test_probe_weights_align_with_diff_of_means(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    belief = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    pr = BeliefProbe(cfg, agent, rng=rng).run(
        corpus, "near_term", belief.directions["near_term"]
    )
    # Convergent validity: learned weights point along the DoM direction.
    assert pr.weight_alignment > 0.3


def test_probe_summary_shape(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    pr = BeliefProbe(cfg, agent, rng=rng).run(corpus, "medium_term")
    s = pr.summary()
    assert s["horizon"] == "medium_term"
    assert 0.0 <= s["best_accuracy"] <= 1.0
