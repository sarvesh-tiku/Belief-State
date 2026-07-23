"""End-to-end smoke test: the full pipeline produces figures + metrics."""

from __future__ import annotations

import os

from beliefstate.data import DataModule
from beliefstate.figures import plot_figure1, plot_figure2
from beliefstate.identify import BeliefStateIdentifier
from beliefstate.intervene import Interventionist
from beliefstate.model import build_agent


def test_full_pipeline_produces_figures(cfg, rng, tmp_path):
    corpus = DataModule(cfg, rng=rng).build()
    agent = build_agent(cfg, rng=rng)
    belief = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    interv = Interventionist(cfg, agent, rng=rng).run(corpus, belief, "near_term")

    out = str(tmp_path)
    fig1 = plot_figure1(belief, out)
    fig2 = plot_figure2(interv, out)

    assert os.path.exists(fig1) and os.path.getsize(fig1) > 0
    assert os.path.exists(fig2) and os.path.getsize(fig2) > 0
    assert belief.passed
    assert interv.directional_specificity > 0
