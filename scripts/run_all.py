"""End-to-end pipeline: data -> identification -> analysis -> interventions.

Usage:
    python -m scripts.run_all [--config config.yaml] [--backend mock|hf]

Runs, in order:
    * Section 3.1   aligned earnings-call / market corpus
    * Section 4     mechanistic identification (3 necessary conditions) + stats
    * Probes        cross-validated linear probes (convergent localization)
    * Section 2.4   non-stationarity stress test (belief-state instability)
    * Section 5     causal activation-level interventions (+ significance)
    * Mediation     Vig-2020 direct/indirect effect decomposition

Produces, under ``results/`` and ``figures/``:
    * results/identification.json, intervention.json, probes.json,
      stress.json, mediation.json
    * figures/figure1.png  (Fig 1 A/B/C)
    * figures/figure2.png  (Fig 2 A/B)
    * figures/figure3.png  (belief-state instability + probe decodability)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np

from beliefstate import load_config
from beliefstate.data import DataModule
from beliefstate.figures import plot_figure1, plot_figure2, plot_figure3
from beliefstate.identify import BeliefStateIdentifier, PatchingDirectionFinder
from beliefstate.intervene import Interventionist
from beliefstate.mediation import MediationAnalysis
from beliefstate.model import build_agent
from beliefstate.probes import BeliefProbe
from beliefstate.stress import StressTest
from beliefstate.tasks import HORIZONS


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serializable: {type(obj)}")


def _dump(results_dir: str, name: str, payload: Any) -> None:
    with open(os.path.join(results_dir, name), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full BELIEF-STATE pipeline.")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--backend", default=None, choices=["mock", "hf"],
                        help="override model.backend")
    parser.add_argument("--horizon", default="near_term",
                        help="horizon for the intervention / stress / mediation sweeps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg.raw["model"]["backend"] = args.backend

    rng = np.random.default_rng(cfg.seed)
    results_dir = cfg.output.results_dir
    figures_dir = cfg.output.figures_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"[belief-state] backend={cfg.model.backend} seed={cfg.seed}")

    # 1) Data (Section 3.1)
    print("[1/6] Building aligned earnings-call / market corpus ...")
    corpus = DataModule(cfg, rng=rng).build()
    n_low = sum(1 for r in corpus if r.regime == "low_vol")
    print(f"      {len(corpus)} aligned tuples "
          f"({n_low} low-vol, {len(corpus) - n_low} high-vol)")

    # 2) Agent backend
    print("[2/6] Loading agent backend ...")
    agent = build_agent(cfg, rng=rng)
    print(f"      hidden_dim={agent.hidden_dim} num_layers={agent.num_layers}")

    # 3) Identification (Section 4) + significance
    print("[3/6] Identifying temporal belief-states ...")
    identifier = BeliefStateIdentifier(cfg, agent, rng=rng)
    belief = identifier.identify(corpus)
    identifier.attach_statistics(belief)
    print("      conditions:", belief.conditions, "-> passed:", belief.passed)
    if belief.significance:
        perm = belief.significance["belief_vs_null_permutation"]
        print(f"      belief>null permutation p={perm['p_value']} "
              f"(Cohen's d={belief.significance['effect_size_cohens_d']})")
    _dump(results_dir, "identification.json", belief.summary())

    # 3b) Cross-validated linear probes (convergent localization)
    print("[4/6] Training cross-validated belief-state probes ...")
    probe = BeliefProbe(cfg, agent, rng=rng)
    probe_results = {
        h: probe.run(corpus, h, belief.directions[h]) for h in HORIZONS
    }
    for h, pr in probe_results.items():
        print(f"      {h}: best layer {pr.best_layer}, "
              f"CV acc {pr.best_accuracy:.3f} (chance {pr.chance:.2f})")
    _dump(results_dir, "probes.json",
          {h: pr.summary() for h, pr in probe_results.items()})

    # 4) Non-stationarity stress test (Section 2.4)
    print("[5/6] Running non-stationarity stress test ...")
    stress = StressTest(cfg, agent, rng=rng).run(corpus, belief, args.horizon)
    print("      stress:", stress.summary())
    _dump(results_dir, "stress.json", stress.summary())

    # 5) Causal interventions (Section 5) + mediation
    print("[6/6] Running causal interventions + mediation ...")
    interv = Interventionist(cfg, agent, rng=rng).run(
        corpus, belief, horizon=args.horizon
    )
    print("      intervention:", interv.summary())
    _dump(results_dir, "intervention.json", interv.summary())

    mediation = MediationAnalysis(cfg, agent, rng=rng).run(
        corpus, belief, args.horizon
    )
    print("      mediation:", mediation.summary())
    _dump(results_dir, "mediation.json", mediation.summary())

    # Causal (activation-patching) direction validation: does the correlational
    # difference-of-means axis coincide with the causally most-effective one?
    finder = PatchingDirectionFinder(cfg, agent, rng=rng).run(
        corpus, belief, args.horizon
    )
    print("      patching direction:", finder.summary())
    _dump(results_dir, "patching_direction.json", finder.summary())

    # Figures
    fig1 = plot_figure1(belief, figures_dir)
    fig2 = plot_figure2(interv, figures_dir)
    fig3 = plot_figure3(stress, probe_results, figures_dir)
    for path in (fig1, fig2, fig3):
        print(f"      wrote {path}")
    print("[done] results/ and figures/ populated.")


if __name__ == "__main__":
    main()
