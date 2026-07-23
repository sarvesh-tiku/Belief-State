"""Real-model backend tests (HFAgent).

These exercise the actual Hugging Face path on a tiny cached model (gpt2), so
they validate the parts the mock backend cannot: correct residual-stream layer
indexing, that steering acts on the layer it claims to, deterministic capture,
non-degenerate reasoning-step token spans, and batch/single equivalence.

The whole module is skipped when torch/transformers or the gpt2 weights are not
available, so the default (mock-only) test run is unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from beliefstate import load_config
from beliefstate.data import DataModule
from beliefstate.model import HFAgent, SpanAlignmentError
from beliefstate.tasks import TaskBuilder

MODEL = "gpt2"


@pytest.fixture(scope="module")
def hf_agent():
    cfg = load_config()
    cfg.raw["model"]["backend"] = "hf"
    cfg.raw["model"]["hf_model_name"] = MODEL
    cfg.raw["model"]["hf_device"] = "cpu"  # deterministic, no MPS nondeterminism
    try:
        agent = HFAgent(cfg)
    except Exception as exc:  # model not downloadable in this environment
        pytest.skip(f"gpt2 unavailable: {exc}")
    return cfg, agent


@pytest.fixture(scope="module")
def a_spec(hf_agent):
    cfg, _ = hf_agent
    cfg = load_config()  # small corpus for prompts
    cfg.raw["data"]["n_firms"] = 4
    cfg.raw["data"]["periods_per_firm"] = 1
    rng = np.random.default_rng(cfg.seed)
    corpus = DataModule(cfg, rng=rng).build()
    builder = TaskBuilder(cfg, rng=rng)
    return builder.render(corpus[0], "near_term", "downside_risk",
                          builder.default_variant())


def test_init_and_shape(hf_agent, a_spec):
    _, agent = hf_agent
    run = agent.run(a_spec)
    # embeddings + one row per block.
    assert run.activations.shape[0] == agent.num_layers + 1
    assert run.activations.shape[1] == a_spec.n_steps
    assert run.activations.shape[2] == agent.hidden_dim


def test_capture_is_deterministic(hf_agent, a_spec):
    _, agent = hf_agent
    a = agent.run(a_spec).activations
    b = agent.run(a_spec).activations
    assert np.allclose(a, b)


def test_steering_targets_the_named_layer(hf_agent, a_spec):
    """Steering layer L must change captured hidden_states[L] and leave L-1
    untouched -- the property the previous off-by-one violated."""

    _, agent = hf_agent
    L = 6
    direction = np.ones(agent.hidden_dim)
    base = agent.run(a_spec).activations
    steered = agent.run_with_steering(a_spec, L, direction, alpha=50.0).activations

    changed = [i for i in range(base.shape[0])
               if not np.allclose(base[i], steered[i], atol=1e-2)]
    assert changed and changed[0] == L
    assert np.allclose(base[L - 1], steered[L - 1], atol=1e-2)


def test_batch_matches_single(hf_agent, a_spec):
    _, agent = hf_agent
    single = agent.run(a_spec).activations
    batched = agent.batch_run([a_spec, a_spec])
    assert len(batched) == 2
    assert np.allclose(batched[0].activations, single, atol=1e-3)


def test_spans_are_non_degenerate(hf_agent, a_spec):
    _, agent = hf_agent
    enc = agent.tokenizer(a_spec.text, return_offsets_mapping=True)
    spans = agent._step_token_spans(a_spec.text, a_spec.n_steps,
                                    enc["offset_mapping"])
    assert len(spans) == a_spec.n_steps
    for s in spans:
        assert len(s) >= 1
    # Steps must not overlap.
    for i in range(1, len(spans)):
        assert spans[i].start >= spans[i - 1].stop


def test_missing_marker_raises(hf_agent):
    _, agent = hf_agent
    text = "no markers here at all"
    enc = agent.tokenizer(text, return_offsets_mapping=True)
    with pytest.raises(SpanAlignmentError):
        agent._step_token_spans(text, 3, enc["offset_mapping"])


def test_run_captures_logits_and_behavioral_readout(hf_agent, a_spec):
    _, agent = hf_agent
    run = agent.run(a_spec)
    assert run.logits is not None
    assert run.logits.shape[0] == agent.model.config.vocab_size
    # Behavioural readout is a finite scalar (risk-minus-calm log-prob contrast).
    val = agent.behavioral_readout(run)
    assert np.isfinite(val)


def test_self_patch_is_identity_hf(hf_agent, a_spec):
    """Patching a layer with its own captured value must reproduce the logits:
    the patch hook overwrites with identical values, so nothing downstream
    changes. This guards the forward-pass patch mechanism."""

    _, agent = hf_agent
    base = agent.run(a_spec)
    L = 6
    target = base.activations[L]  # [n_steps, d]
    patched = agent.run_with_patch(a_spec, L, target)
    assert np.allclose(base.logits, patched.logits, atol=1e-3)


def test_patch_changes_downstream_logits(hf_agent, a_spec):
    """A non-trivial patch at layer L must change the output logits (the change
    propagates through the remaining blocks), unlike a post-hoc tensor edit."""

    _, agent = hf_agent
    base = agent.run(a_spec)
    L = 6
    # Shift along a single direction (not a uniform offset, which the next
    # LayerNorm's mean-centering would remove) on the scale of the steering test.
    bump = np.zeros(agent.hidden_dim, dtype=np.float32)
    bump[0] = 50.0
    target = base.activations[L] + bump[None, :]
    patched = agent.run_with_patch(a_spec, L, target)
    assert not np.allclose(base.logits, patched.logits, atol=1e-2)
