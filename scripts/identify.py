"""Run only the identification pipeline and Figure 1 (Section 4)."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from beliefstate import load_config
from beliefstate.data import DataModule
from beliefstate.figures import plot_figure1
from beliefstate.identify import BeliefStateIdentifier
from beliefstate.model import build_agent
from scripts.run_all import _json_default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default=None, choices=["mock", "hf"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg.raw["model"]["backend"] = args.backend

    rng = np.random.default_rng(cfg.seed)
    corpus = DataModule(cfg, rng=rng).build()
    agent = build_agent(cfg, rng=rng)
    identifier = BeliefStateIdentifier(cfg, agent, rng=rng)
    belief = identifier.identify(corpus)
    identifier.attach_statistics(belief)

    print(json.dumps(belief.summary(), indent=2, default=_json_default))
    os.makedirs(cfg.output.figures_dir, exist_ok=True)
    path = plot_figure1(belief, cfg.output.figures_dir)
    print("wrote", path)


if __name__ == "__main__":
    main()
