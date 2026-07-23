"""Run identification then the causal intervention sweep + Figure 2 (Section 5)."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from beliefstate import load_config
from beliefstate.data import DataModule
from beliefstate.figures import plot_figure2
from beliefstate.identify import BeliefStateIdentifier
from beliefstate.intervene import Interventionist
from beliefstate.model import build_agent
from scripts.run_all import _json_default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default=None, choices=["mock", "hf"])
    parser.add_argument("--horizon", default="near_term")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg.raw["model"]["backend"] = args.backend

    rng = np.random.default_rng(cfg.seed)
    corpus = DataModule(cfg, rng=rng).build()
    agent = build_agent(cfg, rng=rng)
    belief = BeliefStateIdentifier(cfg, agent, rng=rng).identify(corpus)
    interv = Interventionist(cfg, agent, rng=rng).run(
        corpus, belief, horizon=args.horizon
    )

    print(json.dumps(interv.summary(), indent=2, default=_json_default))
    os.makedirs(cfg.output.figures_dir, exist_ok=True)
    path = plot_figure2(interv, cfg.output.figures_dir)
    print("wrote", path)


if __name__ == "__main__":
    main()
