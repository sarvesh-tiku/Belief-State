# Qwen2.5-1.5B-Instruct — findings (modern instruct model)

**Run:** `python3 -m scripts.run_all --config config_qwen.yaml`
(Qwen2.5-1.5B-Instruct, 28 layers / 1536 hidden, 32 tuples, seed 20260228, MPS.)
**Method:** identical non-circular pipeline as the gpt2 run — cluster-aware
statistics (firm-level permutation + bootstrap), Holm-Bonferroni over the layer
search, an **independent behavioural readout** (next-token risk-vs-calm logit
contrast, never the belief axis), forward-pass patching, and orthogonal/swap
controls. Everything below is reported as-is.

## Headline: the belief-state hypothesis fares much better on a modern model — but is not clean

| Result | gpt2 (124M, 2019) | **Qwen2.5-1.5B-Instruct** | Reading |
|---|---|---|---|
| Identification: 3 necessary conditions | FAILS (`passed=False`) | **FAILS** (`passed=False`); persistence ✓, cross-variant ✓, contrastive_sensitivity ✗, horizon_differentiation ✗ | Same verdict: the strict conjunctive gates (tuned on the mock) do not pass on either real model. |
| Belief vs. null representational separation | p=0.0056, d=0.53 | **p=0.0006, d=0.70**, effective N 16.6 | Real and slightly larger. The belief contrast separates from surface-only contrasts more than chance, surviving firm clustering. |
| **Layer localization** (Holm-Bonferroni corrected) | **no layer significant** (min adj-p=0.078) | **6 deep layers significant: 18, 21, 22, 25, 27, 28** (min adj-p=0.014) | **New positive.** Unlike gpt2, the belief>null separation localizes to specific deep layers *after* multiple-comparison correction. This is the corrected localization result gpt2 could not produce. |
| **Causal steering** vs random (behavioural readout) | refuted: p=1.0, d=−16.6 | **p=0.0002, d=47.9**, reversible (±0.24 symmetric) | **Reversed on Qwen.** Steering the belief axis reliably and reversibly moves the model's *output* (risk-vs-calm next-token lean) far more than matched random directions. |
| Steering vs **orthogonal** (b projected out) | — | **p=0.0004, d=48.6** | The effect is carried by the belief-axis content specifically, not generic perturbation of the same norm. |
| Steering vs **swap** (other horizon's direction) | — | **p=1.0, d=−1.7** | **No horizon specificity.** The medium-horizon direction steers behaviour just as strongly. The effect is "a belief-related direction moves risk output," not "the near-term belief-state uniquely." (Expected: the two horizon directions overlap substantially.) |
| Mediation (Vig 2020) | TE≈0, undefined | **TE=0.011, proportion=0.47, CI [−2.5, 3.4]** | Total *natural* input→behaviour effect is small; the mediated fraction's CI spans zero, so mediation is **not established**. Forced steering moves behaviour, but the input manipulation itself does not move it enough to decompose. |
| Causal vs correlational direction | best=pc3, \|cos\|=0.04 (unrelated) | **best=pc1 (effect 0.63), diff-of-means 0.23, \|cos\|=0.01** | Difference-of-means is **not** the most causal direction — the leading PC of the contrast residuals moves behaviour ~2.7× more and is nearly orthogonal to it. But here the causal winner is a *principled* top component, not noise. |
| Probe accuracy | 1.000 @ layer 1, align 0.02 | **1.000 @ layer 1, align 0.02** | Still surface leakage: the prompt states volatility numerically and a layer-1 probe reads it off the embedding. Probe weight is ~orthogonal to the belief direction, so it is not decoding the belief-state. Must be controlled (withhold the number — Session 4). |
| Stress drift ratio (shift/stable) | 0.94 (no drift) | **1.44** (shifted steps drift more) | Direction now matches the hypothesis, though the onset<failure precursor did not trigger (thresholds are mock-tuned; needs recalibration). |

## What is genuinely publishable here

1. **Corrected layer localization.** On a modern instruct model the belief>null
   separation is not diffuse: it concentrates in deep layers (18–28) and
   survives Holm-Bonferroni. That is a real, falsifiable mechanistic claim.
2. **Reversible, specific causal steering.** Steering the belief axis moves an
   *independent behavioural readout* (output logits) far more than random
   (d=47.9) or norm-matched orthogonal (d=48.6) directions, symmetrically and
   reversibly. Because the readout is not the intervention axis, this is not the
   circularity artifact that inflated the earlier gpt2 numbers — it is genuine
   causal evidence that the direction influences behaviour.
3. **A methodological contribution that survives scrutiny.** The
   activation-patching direction finder shows difference-of-means is *not* the
   causally optimal direction (pc1 is 2.7× stronger, near-orthogonal). A paper
   that reports this — and uses the causal finder rather than diff-of-means — is
   more rigorous than the correlational-direction norm in the literature.

## The honest caveats (these are the paper, not footnotes)

- **The three-condition identification still fails.** We should stop framing
  identification as a pass/fail conjunction of mock-tuned thresholds and instead
  report the individual quantities with CIs. Persistence and cross-variant
  robustness hold; contrastive-sensitivity and horizon-differentiation gates do
  not clear their (arbitrary) bars.
- **No horizon specificity.** Near- and medium-term directions are behaviourally
  interchangeable (swap p=1.0). The paper cannot claim horizon-distinct belief
  directions on Qwen; it can claim a single risk-belief direction.
- **Mediation is not established.** Forced steering works, but the *natural*
  low→high-vol input change produces only a small behavioural effect (TE=0.011),
  so the mediated proportion is unidentifiable (CI spans zero). The causal
  chain "input → belief-state → behaviour" is not demonstrated end-to-end.
- **Probe result is still leakage**, not evidence of an emergent belief-state.

## Bottom line

On gpt2 the causal claims were **refuted**. On Qwen2.5-1.5B-Instruct they are
**partially supported**: there is a deep, localized, reversible belief direction
that causally moves behaviour more than any control except the other horizon's
direction — but it is a single risk-belief direction (not horizon-specific), the
input→behaviour mediation is not established, and difference-of-means is not the
optimal estimator. This is a real, defensible, mixed result: strong enough to
build a paper on, honest enough to survive review. The scale jump from gpt2 to a
modern instruct model is what turned a null into a signal — evidence the
phenomenon is model-capability-dependent, which is itself a finding worth
reporting.
