<div align="center">

# BELIEF-STATE

### Interpreting Temporal Belief Dynamics in Agentic Financial Systems

[![Venue](https://img.shields.io/badge/ICLR%202026-Advances%20in%20Financial%20AI-8A2BE2)](https://iclr.cc/)
[![License](https://img.shields.io/badge/license-MIT-black)](LICENSE)

*Sarvesh Tiku* · ICLR 2026 Workshop on Advances in Financial AI

</div>

An agent's forecast about a market may be present in its activations before it
surfaces in any output token. This work asks a falsifiable version of that
question: **is there a linear direction in the residual stream that encodes an
agent's expected future over a fixed horizon, and does steering it causally move
behavior?** We call such a direction a *temporal belief-state*. We build a
pipeline that (i) recovers candidate directions by contrast, (ii) tests them
against surface-only null contrasts, (iii) steers them and reads out an
**independent** behavioral channel, and (iv) tracks their drift under regime
shift. We run it on a synthetic model with a known ground truth (to validate the
machinery) and on two real language models (to test the hypothesis). The results
are reported as-is, including where they contradict the hypothesis.

> **Status: a mixed, honestly-reported result — not a clean confirmation.** On
> GPT-2 the causal claims are **refuted**. On Qwen2.5-1.5B-Instruct they are
> **partially supported**: a localized, reversible belief direction causally
> moves behavior beyond random and orthogonal controls, but it is not
> horizon-specific and end-to-end mediation is not established. This README and
> the `results_*/` writeups reflect exactly what the code produces.

## Why this is a methods contribution first

The natural way to test a "belief direction" is circular: project activations
onto the direction, steer along it, then read the projection back out. That
guarantees a large effect regardless of whether the direction means anything. An
earlier version of this pipeline reported Cohen's *d* ≈ 118 that way — an
artifact. This repo is built to **avoid** that:

- **Independent behavioral readout.** Steering is scored against a downstream
  next-token *risk-vs-calm* logit contrast that never touches the belief axis.
  Mediator ≠ readout.
- **Forward-pass patching.** `run_with_patch` replaces a residual at a layer
  *during* the forward pass (delta-based, so a self-patch is exactly identity),
  letting edits propagate to the logits.
- **Specificity controls.** Every steering claim is compared against
  norm-matched random directions, an orthogonal-subspace control (the belief
  content projected out), and a swap control (the other horizon's direction).
- **Cluster-aware statistics.** Contrast pairs and steering instances are
  correlated within firm, so p-values come from firm-level cluster permutation
  tests, CIs from a firm-level cluster bootstrap, effective *N* is design-effect
  corrected, and the layerwise search is Holm-Bonferroni corrected. Cohen's *d*
  is uncapped and returns NaN on degenerate SD (`stats.py`).
- **An activation-patching direction finder** that scores candidate directions
  (difference-of-means and top PCs) by their *causal* effect on the behavioral
  readout — and reports whether the correlational direction is the causal one.

Validating on a synthetic model with a planted ground truth shows the machinery
works (steering moves behavior monotonically, mediation proportion ≈ 0.98, the
finder recovers difference-of-means with alignment 1.0). That is what lets us
trust the real-model negatives as real, not code artifacts.

## What the code finds

| Claim | Mock (ground truth) | GPT-2 (124M) | Qwen2.5-1.5B-Instruct |
|---|---|---|---|
| Identification: 3 necessary conditions | pass (by construction) | **fail** | **fail** (persistence ✓, cross-variant ✓) |
| Belief > null representational separation | d≈4.5 | p=0.0056, d=0.53 | **p=0.0006, d=0.70** |
| Layer localization (Holm-Bonferroni) | sharp peak | **none survives** | **6 deep layers survive** (min adj-p=0.014) |
| Causal steering vs random (independent readout) | moves it | **refuted** (p=1.0) | **p=0.0002, d=47.9, reversible** |
| Steering vs orthogonal control | — | negligible | **p=0.0004, d=48.6** |
| Steering vs swap (other horizon) | — | n.s. | **n.s. (p=1.0) — no horizon specificity** |
| Mediation (Vig 2020), input→belief→behavior | 0.98, CI excludes 0 | TE≈0, undefined | prop=0.47, **CI [−2.5, 3.4] spans 0** |
| Causal vs correlational direction | aligned (1.0) | orthogonal (\|cos\|=0.04) | orthogonal (\|cos\|=0.01) |
| Probe accuracy | 0.99 | 1.000 @ layer 1 | 1.000 @ layer 1 |

**Reading the table.**

- **Identification.** The three-condition conjunction (tuned on the mock) does
  not pass on either real model. Persistence and cross-variant robustness hold;
  contrastive-sensitivity and horizon-differentiation do not clear their bars.
  The individual quantities, with CIs, are the honest object — not a pass/fail
  flag.
- **Localization is the strongest positive.** On Qwen the belief > null
  separation concentrates in deep layers (18, 21, 22, 25, 27, 28) and survives
  multiple-comparison correction. GPT-2 has no surviving layer. This is a
  genuine, falsifiable mechanistic claim on a modern model.
- **Causal steering reverses with scale.** On GPT-2, steering the belief axis
  moves the independent readout *less* than random directions (refuted). On
  Qwen, it moves it far more than random (d=47.9) *and* than norm-matched
  orthogonal directions (d=48.6), symmetrically and reversibly — genuine causal
  evidence, because the readout is not the intervention axis.
- **But it is one risk-belief direction, not horizon-distinct.** The other
  horizon's direction steers just as well (swap p=1.0), so the paper claims *a*
  belief direction, not per-horizon belief-states.
- **Mediation is not established.** Forced steering works, but the *natural*
  low→high-volatility input change produces too small a total effect (TE=0.011)
  to decompose; the mediated proportion's CI spans zero.
- **Correlational ≠ causal.** Difference-of-means is not the causally optimal
  direction on either model; the leading contrast PC moves behavior more and is
  nearly orthogonal to it. The direction finder is the right tool, not
  difference-of-means alone.
- **The probe result is surface leakage**, not evidence of an emergent belief:
  the prompt states volatility numerically and a layer-1 probe reads it off the
  embedding (weight ≈ orthogonal to the belief axis). Withholding the number is
  the next control.

Full per-model writeups: [`results_hf/COMPARISON.md`](results_hf/COMPARISON.md)
(GPT-2) and [`results_qwen/FINDINGS.md`](results_qwen/FINDINGS.md) (Qwen).

## Reproduce

```bash
python3 -m pip install -r requirements.txt

# 1. Infrastructure validation — synthetic model with a known belief-state.
python3 -m scripts.run_all                          # writes results/, figures/

# 2. Empirical test on real models.
python3 -m scripts.run_all --config config_hf.yaml    # GPT-2  → results_hf/,  figures_hf/
python3 -m scripts.run_all --config config_qwen.yaml  # Qwen   → results_qwen/, figures_qwen/
```

The mock run is offline and finishes in under a minute. Each stage also runs
standalone (`scripts.build_data`, `scripts.identify`, `scripts.intervene`).
Tests: `pytest -q` — **65 tests**, including ground-truth recovery and
self-patch-is-identity checks on both the mock and HF backends. The HF-backend
tests are skipped automatically when torch/transformers or the weights are
unavailable.

**Backends.** The default `mock` backend is a synthetic transformer with an
analytically known belief-state, used only to validate that the pipeline
recovers a planted ground truth. It is **infrastructure validation, not evidence
about language models** — the two are kept strictly separate throughout. The
`hf` backend swaps in real activations via forward hooks on the residual stream
(the same hook applies steering and patching) and changes nothing else; any
HuggingFace causal-LM works via `hf_model_name` / `hf_block_path`.

`config.yaml` (mock), `config_hf.yaml` (GPT-2), and `config_qwen.yaml` (Qwen)
hold the seed, thresholds, α-grid, and data split for each run.

## Layout

```
beliefstate/   data · tasks · model            corpus, prompts, activations, patching
               identify · intervene            contrasts + conditions, steering + controls
               probes · stress · mediation     grouped probes, drift, Vig (2020) mediation
               stats · figures                 cluster-aware inference, figures
scripts/       run_all + per-stage entry points
tests/         65 unit + end-to-end tests
results_hf/    GPT-2 outputs + COMPARISON.md    results_qwen/  Qwen outputs + FINDINGS.md
```

## Limitations and next steps

- **Surface leakage.** The volatility regime is stated numerically in the
  prompt; identification and probing must be re-run with the number withheld to
  measure an *inferred* belief rather than a copied input.
- **Horizon specificity is absent** on Qwen (swap p=1.0). Either the horizons
  are not representationally distinct, or the contrast does not separate them —
  an open question.
- **Mediation is underpowered** because the natural input effect is small; a
  stronger treatment or a more sensitive behavioral readout is needed.
- **Synthetic financial data.** The corpus is generated; real filings/price
  series are future work.
- Results are single-seed per model; multi-seed CIs would strengthen the
  representational claims.

## Citation

```bibtex
@inproceedings{tiku2026beliefstate,
  title     = {{BELIEF-STATE}: Interpreting Temporal Belief Dynamics in Agentic Financial Systems},
  author    = {Tiku, Sarvesh},
  booktitle = {ICLR 2026 Workshop on Advances in Financial AI},
  year      = {2026}
}
```

Causal mediation follows Vig et al., *Investigating Gender Bias in Language
Models Using Causal Mediation Analysis*, NeurIPS 2020.
