"""Build (or load) the aligned corpus and print a summary (Section 3.1)."""

from __future__ import annotations

import argparse

import numpy as np

from beliefstate import load_config
from beliefstate.data import DataModule, split_by_regime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus = DataModule(cfg, rng=np.random.default_rng(cfg.seed)).build()
    by_regime = split_by_regime(corpus)

    print(f"aligned tuples: {len(corpus)}")
    for regime, recs in by_regime.items():
        vols = np.array([r.hist_vol * np.sqrt(252) for r in recs])
        rvols = np.array([r.realized_vol * np.sqrt(252) for r in recs])
        print(f"  {regime:9s} n={len(recs):3d}  "
              f"hist ann-vol mean={vols.mean() * 100:5.1f}%  "
              f"realized ann-vol mean={rvols.mean() * 100:5.1f}%")

    sample = corpus[0]
    print("\n--- sample aligned tuple ---")
    print("ticker:", sample.ticker, "sector:", sample.sector,
          "regime:", sample.regime, "uncertainty:", sample.uncertainty)
    print("market context:", sample.market_context_text())
    print("transcript:", sample.transcript)


if __name__ == "__main__":
    main()
