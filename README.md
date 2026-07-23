<div align="center">

# BELIEF-STATE

### Interpreting Temporal Belief Dynamics in Agentic Financial Systems

[![Venue](https://img.shields.io/badge/ICLR%202026-Advances%20in%20Financial%20AI-8A2BE2)](https://iclr.cc/)
[![License](https://img.shields.io/badge/license-MIT-black)](LICENSE)

*Sarvesh Tiku* · ICLR 2026 Workshop on Advances in Financial AI

</div>

An agent's forecast about a market lives in its activations before it appears in any output token. This work locates that forecast. A **temporal belief-state** is a linear direction in the residual stream that encodes the agent's expected future over a fixed horizon. We recover it by contrast, show it is decodable, steerable, and mediating, and show that its instability under regime shift is a leading indicator of failure. This repo runs the full pipeline from one seed.

## What the code establishes

**Identification (§4).** `identify.py` recovers per-horizon belief directions by difference-of-means over controlled contrasts (vary history, hold disclosure fixed, and the reverse). It verifies the three necessary conditions — contrastive sensitivity, temporal persistence, horizon differentiation — plus cross-variant robustness (§4.2): the direction is stable under paraphrase, reordering, and reformatting, whereas surface-only null contrasts produce no such direction. Each horizon localizes to a single layer.

**Probes.** `probes.py` fits grouped k-fold logistic probes at every layer, grouping by task instance so no reasoning step leaks across folds. Peak probe accuracy coincides with the difference-of-means peak, and probe weights align with that axis — an independent localization rather than a restatement of the contrast.

**Intervention (§5).** `intervene.py` applies `h' = h + αb` across an α-grid with norm-matched random-direction controls. Behavioral response is monotone in α, sign-reversible, and specific to **b** over random directions. `mediation.py` decomposes the effect of a low→high volatility treatment via activation patching (Vig et al., 2020): patching the belief-state to its counterfactual value accounts for the treatment's effect on behavior; the direct path does not.

**Stress (§2.4).** `stress.py` measures per-step drift of the belief-state projection under a sustained regime shift against a matched stable condition. Drift onset precedes the operationalized failure step and is visible at the representation level before any output error.

Permutation tests, bootstrap CIs, and Cohen's *d* (`stats.py`) are attached to the comparative claims.

## Reproduce

```bash
python3 -m pip install -r requirements.txt
python3 -m scripts.run_all          # data → identify → probe → stress → intervene → mediate
```

Runs offline in under a minute and writes `results/*.json` and the figures. For a real model, set `model.backend: hf` (or `--backend hf`) with any HuggingFace causal-LM; forward hooks read the residual stream and the same hook applies the steering term. Stages also run standalone (`scripts.build_data`, `scripts.identify`, `scripts.intervene`). Tests: `pytest -q` (42, including ground-truth recovery on the mock backend).

`config.yaml` holds the seed, thresholds, α-grid, and data split. The default `mock` backend is a synthetic transformer with an analytically known belief-state, so the identification and steering results have a ground truth to check against; `hf` swaps in real activations and changes nothing else.

## Layout

```
beliefstate/   data · tasks · model            corpus, prompts, activations
               identify · intervene            §4 conditions, §5 steering
               probes · stress · mediation     probes, §2.4 drift, causal mediation
               stats · figures                 inference, figures
scripts/       run_all + per-stage entry points
tests/         42 unit + end-to-end tests       config.yaml — seed, thresholds, splits
```

## Citation

```bibtex
@inproceedings{tiku2026beliefstate,
  title     = {{BELIEF-STATE}: Interpreting Temporal Belief Dynamics in Agentic Financial Systems},
  author    = {Tiku, Sarvesh},
  booktitle = {ICLR 2026 Workshop on Advances in Financial AI},
  year      = {2026}
}
```

Causal mediation follows Vig et al., *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*, NeurIPS 2020.
