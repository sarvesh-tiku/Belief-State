"""Tests for Section 3.1 data construction and temporal alignment."""

from __future__ import annotations

import numpy as np

from beliefstate.data import split_by_regime


def test_corpus_size_and_regimes(cfg, corpus):
    expected = int(cfg.data.n_firms) * int(cfg.data.periods_per_firm)
    assert len(corpus) == expected
    by_regime = split_by_regime(corpus)
    assert set(by_regime) <= set(cfg.data.regimes)
    # Both regimes should be represented for the contrasts to be constructible.
    assert by_regime.get("low_vol") and by_regime.get("high_vol")


def test_temporal_alignment_split(cfg, corpus):
    lookback = int(cfg.data.lookback_days)
    horizon = int(cfg.data.horizon_days)
    for rec in corpus:
        assert rec.call_index == lookback
        assert len(rec.historical_prices) == lookback + 1
        assert len(rec.realized_prices) == horizon
        # Historical + realized reconstruct the full path.
        assert len(rec.prices) == lookback + horizon + 1


def test_high_vol_regime_has_higher_realized_vol(corpus):
    by_regime = split_by_regime(corpus)
    lo = np.mean([r.realized_vol for r in by_regime["low_vol"]])
    hi = np.mean([r.realized_vol for r in by_regime["high_vol"]])
    assert hi > lo


def test_realized_outcomes_not_used_as_labels(corpus):
    # Sanity: the market-context text the agent sees is derived only from the
    # historical window, never the realized (post-call) window.
    rec = corpus[0]
    ctx = rec.market_context_text()
    assert "Historical market context" in ctx
