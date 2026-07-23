"""Tests for the Session 3 causal machinery (model patching + readout,
activation-patching direction finder).

These pin the properties that make the causal analysis *non-circular*:

  * the behavioural readout is independent of the belief axis but still moves
    when the belief axis is steered (through the model's own computation);
  * ``run_with_patch`` replaces the residual at a layer and propagates the
    change downstream, and a self-patch is exactly the identity;
  * the activation-patching direction finder recovers the difference-of-means
    axis on the mock (where it *is* the causal axis by construction).

They run on the mock backend only, so no model download is required.
"""

from __future__ import annotations

import numpy as np

from beliefstate.identify import BeliefStateIdentifier, PatchingDirectionFinder
from beliefstate.model import MockAgent
from beliefstate.tasks import TaskBuilder


def _agent_and_spec(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    builder = TaskBuilder(cfg, rng=rng)
    spec = builder.render(corpus[0], "near_term", "downside_risk",
                          builder.default_variant())
    return agent, spec


def test_self_patch_is_identity(cfg, corpus, rng):
    agent, spec = _agent_and_spec(cfg, corpus, rng)
    run = agent.run(spec)
    L = agent.peak_layer("near_term")
    patched = agent.run_with_patch(spec, L, run.activations[L])
    assert np.allclose(patched.activations, run.activations, atol=1e-9)
    assert np.isclose(agent.behavioral_readout(patched),
                      agent.behavioral_readout(run), atol=1e-9)


def test_patch_propagates_downstream(cfg, corpus, rng):
    agent, spec = _agent_and_spec(cfg, corpus, rng)
    run = agent.run(spec)
    L = agent.peak_layer("near_term")
    # Patch the belief content up at the peak layer -> later layers must change.
    b = agent.true_direction("near_term")
    target = run.activations[L] + 3.0 * b[None, :]
    patched = agent.run_with_patch(spec, L, target)
    assert not np.allclose(patched.activations[L + 1], run.activations[L + 1])
    # And the independent behavioural readout must move.
    assert not np.isclose(agent.behavioral_readout(patched),
                          agent.behavioral_readout(run))


def test_behavioral_readout_independent_but_responsive(cfg, corpus, rng):
    """The readout axis is not the belief axis (independence), yet steering the
    belief axis moves the readout monotonically (responsiveness through the
    model), which is exactly the non-circular property mediation relies on."""

    agent, spec = _agent_and_spec(cfg, corpus, rng)
    L = agent.peak_layer("near_term")
    b = agent.true_direction("near_term")
    responses = [
        agent.behavioral_readout(agent.run_with_steering(spec, L, b, a))
        for a in (-4.0, -2.0, 0.0, 2.0, 4.0)
    ]
    assert all(np.diff(responses) > 0)  # strictly increasing
    # The behaviour axis is not identical to the belief axis.
    assert abs(float(agent._w_behavior @ b)) < 0.99


def test_orthogonal_steer_does_not_move_behavior(cfg, corpus, rng):
    """A steer orthogonal to the belief axis carries no belief content, so on the
    mock (where behaviour is driven only through the belief axis) it must not
    move the readout -- the specificity control's key property."""

    agent, spec = _agent_and_spec(cfg, corpus, rng)
    L = agent.peak_layer("near_term")
    b = agent.true_direction("near_term")
    base = agent.behavioral_readout(agent.run(spec))
    v = rng.standard_normal(agent.hidden_dim)
    v = v - (v @ b) * b
    v = v / np.linalg.norm(v)
    moved = agent.behavioral_readout(agent.run_with_steering(spec, L, v, 5.0))
    assert np.isclose(moved, base, atol=1e-6)


def test_patching_finder_recovers_diff_of_means_on_mock(cfg, corpus, rng):
    agent = MockAgent(cfg, rng=rng)
    belief = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    res = PatchingDirectionFinder(cfg, agent, rng=rng).run(
        corpus, belief, "near_term"
    )
    # On the mock the difference-of-means axis IS the causal axis.
    assert res.best_label == "diff_of_means"
    assert res.best_vs_diff_alignment > 0.99
    # Off-belief principal components have negligible causal effect.
    effects = dict(zip(res.candidate_labels, res.causal_effects))
    assert effects["diff_of_means"] > 10 * effects.get("pc2", 0.0)
