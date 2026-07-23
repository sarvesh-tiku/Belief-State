"""Tests for Sections 3.2/3.3/4.2: tasks, prompts, contrasts, variants."""

from __future__ import annotations

from beliefstate.data import split_by_regime
from beliefstate.tasks import TaskBuilder


def test_controlled_history_contrast_shares_disclosure(cfg, corpus):
    builder = TaskBuilder(cfg)
    by_regime = split_by_regime(corpus)
    lo, hi = by_regime["low_vol"][0], by_regime["high_vol"][0]
    variant = builder.default_variant()
    a, b = builder.history_contrast(lo, hi, "near_term", "downside_risk", variant)
    # The disclosure is held fixed; only the market context differs.
    assert a.disclosure == b.disclosure
    assert a.market_context != b.market_context


def test_controlled_disclosure_contrast_shares_history(cfg, corpus):
    builder = TaskBuilder(cfg)
    rec = corpus[0]
    variant = builder.default_variant()
    a, b = builder.disclosure_contrast(
        rec, "guidance reaffirmed", "guidance withdrawn",
        "near_term", "downside_risk", variant,
    )
    assert a.market_context == b.market_context
    assert a.disclosure != b.disclosure


def test_null_contrasts_preserve_semantics(cfg, corpus):
    builder = TaskBuilder(cfg)
    rec = corpus[0]
    variant = builder.default_variant()
    for maker in (
        builder.null_contrast_reorder,
        builder.null_contrast_paraphrase,
        builder.null_contrast_format,
    ):
        a, b = maker(rec, "near_term", "downside_risk", variant)
        # Same regime/uncertainty semantics on both sides of a null contrast.
        assert a.regime == b.regime
        assert a.uncertainty == b.uncertainty


def test_variant_grid_is_nonempty_and_unique(cfg):
    builder = TaskBuilder(cfg)
    variants = builder.all_variants()
    keys = [v.key() for v in variants]
    assert len(keys) == len(set(keys))
    assert len(keys) > 1


def test_reasoning_step_markers_present(cfg, corpus):
    builder = TaskBuilder(cfg)
    spec = builder.render(corpus[0], "near_term", "downside_risk",
                          builder.default_variant())
    for i in range(spec.n_steps):
        assert builder.render_marker(i) in spec.text
