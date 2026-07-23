"""Tests for Section 5: causal interventions on belief-states."""

from __future__ import annotations

import numpy as np

from beliefstate.identify import BeliefStateIdentifier
from beliefstate.intervene import Interventionist
from beliefstate.model import MockAgent


def _run(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    belief = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    interv = Interventionist(cfg, agent, rng=rng).run(corpus, belief, "near_term")
    return interv


def test_belief_response_is_monotonic_in_alpha(cfg, corpus, rng):
    interv = _run(cfg, corpus, rng)
    order = np.argsort(interv.alpha_grid)
    resp = interv.belief_response[order]
    # Non-decreasing (allowing tiny numerical noise) response to steering.
    assert np.all(np.diff(resp) > -1e-6)


def test_zero_alpha_gives_near_zero_response(cfg, corpus, rng):
    interv = _run(cfg, corpus, rng)
    zero_idx = int(np.argmin(np.abs(interv.alpha_grid)))
    assert abs(interv.belief_response[zero_idx]) < 1e-6


def test_reversibility_positive_and_negative(cfg, corpus, rng):
    interv = _run(cfg, corpus, rng)
    rev = interv.reversibility
    # Negative steering lowers, positive steering raises, symmetric-ish.
    assert rev["belief_neg"] < 0 < rev["belief_pos"]
    assert np.isclose(rev["belief_pos"], -rev["belief_neg"], atol=0.15)


def test_random_directions_have_no_systematic_effect(cfg, corpus, rng):
    interv = _run(cfg, corpus, rng)
    # Random-direction control response stays near zero across the grid.
    assert np.max(np.abs(interv.random_response)) < 0.2


def test_directional_specificity_positive(cfg, corpus, rng):
    interv = _run(cfg, corpus, rng)
    # Belief steering moves the *behavioural* readout more than random directions.
    assert interv.directional_specificity > 0


def test_belief_beats_orthogonal_and_swap_behaviorally(cfg, corpus, rng):
    """The specificity controls: an orthogonal (belief projected out) steer
    carries no belief content and must not move behaviour, and the other
    horizon's direction (swap) must move it less than the matched belief axis."""

    interv = _run(cfg, corpus, rng)
    span = lambda a: float(np.max(a) - np.min(a))
    belief_span = span(interv.belief_response)
    assert belief_span > span(interv.orthogonal_response)
    assert belief_span >= span(interv.swap_response)


def test_projection_diagnostic_is_not_the_behavioral_outcome(cfg, corpus, rng):
    """The mechanistic belief-projection diagnostic moves ~linearly with alpha
    (it re-reads the intervention axis); the behavioural outcome is a distinct,
    smaller-scale quantity. They must not be identical."""

    interv = _run(cfg, corpus, rng)
    assert not np.allclose(
        interv.belief_projection_response, interv.belief_response, atol=1e-6
    )
    # The projection diagnostic swing is (much) larger than the behavioural one.
    span = lambda a: float(np.max(a) - np.min(a))
    assert span(interv.belief_projection_response) > span(interv.belief_response)
