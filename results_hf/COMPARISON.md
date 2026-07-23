# Mock vs. real-model (gpt2) — honest comparison

**Run:** `python3 -m scripts.run_all --config config_hf.yaml` (gpt2, 32 tuples, seed 20260228, ~2 min on CPU/MPS).
**Purpose:** separate *infrastructure validation* (the pipeline runs and is analytically correct on the mock) from *empirical evidence* (what a real model actually does). Numbers below are reported as-is, including the parts that undercut the paper's current claims.

All gpt2 statistics below are **cluster-aware**: contrast pairs and steering
instances are correlated within firm, so p-values come from a firm-level cluster
permutation test, CIs from a firm-level cluster bootstrap, and we report the
design-effect-corrected effective N. The per-layer localization search is
Holm-Bonferroni-corrected across the layer family. Cohen's d is uncapped and
returns NaN when the pooled SD is degenerate.

| Claim | Mock (config.yaml) | gpt2 (config_hf.yaml) | Reading |
|---|---|---|---|
| Identification: 3 conditions | **all pass** | **FAILS** — contrastive_sensitivity ✗, horizon_differentiation ✗ (persistence ✓, cross-variant ✓); `passed=False` | The mock passes by construction. gpt2 does **not** satisfy the necessary conditions under these thresholds. |
| Belief vs. null separation | p=2e-4, d=4.49 | **p=0.0056** (cluster perm), **d=0.53**; N 32→**16.4** effective | A real but *small* effect that survives firm clustering. Effective N drops from 32 to 16.4 (two contrast kinds per firm), yet belief>null holds. Not the mock's d≈4.5. |
| Layer localization (peak layer) | sharp peak @ L13 | **no layer significant** after Holm-Bonferroni (min adjusted p=0.078) | **New negative result.** The argmax "peak layer" is picked from a layerwise search; once corrected for that search, **no single layer** shows belief>null localization at α=0.05. The peak-layer claim does not survive multiple-comparison correction on gpt2. |
| Probe accuracy | 0.99 @ L13/16 | **1.000 @ layer 1** | **Surface leakage, not evidence.** The prompt states volatility numerically; a layer-1 probe reads it off the embedding. This is a *negative* result for "emergent belief-state" and must be controlled (see below). |
| Intervention — *circular* projection readout | 1.97× | spec=8.87, p=2e-4, **d=118** | The readout **is** the intervention axis, so this is mechanical, not behavioural. Retained only to show what the circular metric reports. |
| Intervention — **independent behavioural readout** (Session 3) | belief moves it, spec>0 | **belief vs random p=1.0, d=−16.6**; **belief vs swap p=0.26** | **The causal claim is refuted on gpt2.** Steering the belief axis moves the model's *output* (risk-vs-calm logit contrast) **less** than random directions and no more than the other horizon's axis. It survives only against the orthogonal control (p=0.0012, tiny d=1.5). |
| Mediation proportion (Vig 2020) | 0.98, CI [0.96,1.01] | **TE≈0.0008, proportion=3.9, CI [−35,+32]** | With a forward-pass patch + independent readout, the belief-state mediates **no** behavioural effect: total effect is ~0 and the proportion is a meaningless ratio of near-zero numbers (CI spans zero by a wide margin). Not the earlier 1.0. |
| Causal vs correlational direction (Session 3) | best=diff_of_means, align=1.0 | **best=pc3, align=0.036** | The causally most-effective patch direction is nearly **orthogonal** to difference-of-means. The correlational direction is not the causal one on gpt2. |
| Stress: onset < failure | True (3<6) | **False** (2==2), drift ratio 0.94 | With the injected perturbation removed by the real forward pass, gpt2 shows **no natural pre-failure drift** (ratio < 1). The stress narrative does not hold on a real model as currently measured. |

## What this establishes

1. **The pipeline is now real.** The HF backend runs end to end on gpt2 with correct layer indexing (steering layer L modifies exactly `hidden_states[L]`, verified in `tests/test_hf_backend.py`), robust span alignment, batching, and determinism.
2. **The mock's results do not transfer.** Identification fails, the effect size collapses from d≈4.5 to d≈0.5, and the stress precursor disappears. This is the expected consequence of removing the planted ground truth — and it is the honest baseline the paper must build on.
3. **The corrected statistics remove one surviving claim and expose two artifacts:**
   - **Layer localization does not survive multiple-comparison correction.** The "peak layer" is the argmax of a layerwise search; after Holm-Bonferroni across that family, no layer separates belief from null at α=0.05 (min adjusted p=0.078). The paper cannot claim a localized belief layer on gpt2.
   - **Firm clustering roughly halves the identification evidence** (effective N 32→16.4) but the belief>null separation still holds (p=0.0056). This is the one real, if small, positive signal.
   - Probe 1.000 @ layer 1 = the regime is stated in the prompt text. Needs a **surface-only null probe** and a version where the numeric volatility is withheld from the prompt.
   - Intervention (d≈118 uncapped, effective N unchanged by clustering) and mediation ≈ 1.0 both use the belief projection as *both* the intervention axis and the readout, so they are circular. The near-constant per-firm gap is what drives the enormous d. Needs a **behavioral readout** (generated-text metric) independent of the projection.
4. **Session 3 supplies that independent readout — and the causal story does not survive it.** With the behavioural channel (next-token risk-vs-calm logit contrast) and forward-pass patching:
   - **Steering is not causally specific.** Belief-axis steering moves gpt2's *output* less than matched random directions (p=1.0, d=−16.6) and no more than the other horizon's direction (swap p=0.26). It beats only the orthogonal control (p=0.0012) by a negligible margin (d=1.5).
   - **The belief-state does not mediate behaviour.** Total effect of the low→high-vol treatment on the behavioural readout is ~0 (0.0008), so the Vig decomposition is undefined (proportion=3.9, CI [−35,+32] spanning zero).
   - **Correlational ≠ causal direction.** The activation-patching finder's best direction (pc3) is nearly orthogonal to difference-of-means (|cos|=0.036). Difference-of-means finds *what changes with the input*, not *what changes behaviour*.
   - **Why the old numbers looked strong:** the previous d≈118 / proportion≈1.0 were entirely the circularity artifact (readout = intervention axis). Removing it removes the effect. This is the single most important correction in the rewrite.

## Immediate consequences for the next sessions

- **Session 2 (stats): DONE.** All gpt2 statistics are now cluster-aware (firm-level permutation + bootstrap), the layer search is Holm-Bonferroni-corrected, and Cohen's d is uncapped. Result: the peak-layer localization claim does **not** survive correction; the belief>null separation does (p=0.0056, effective N 16.4).
- **Session 3 (causal methods): DONE.** Added an independent behavioural readout (output-logit contrast, never the belief axis), forward-pass patching (`run_with_patch`), orthogonal-subspace + swap controls, an activation-patching direction finder, and full-vector Vig mediation. Result: **the causal claims on gpt2 are refuted** — steering is not behaviourally specific (p=1.0 vs random), the belief-state mediates ~0 behavioural effect, and the causal direction is orthogonal to difference-of-means. The mock still validates the machinery (steering moves behaviour, proportion mediated 0.98), confirming the negative gpt2 result is real, not a code artifact.
- **Session 4 (behavioral grounding + stress):** measure *natural* drift (already de-injected here — currently ~0, report honestly); withhold the volatility number from prompts to test whether an *inferred* belief exists at all once the leakage is gone.
- **Data:** withhold the literal volatility number from the prompt (or add a leakage control) so identification/probing measures an *inferred* belief, not a copied input.

Bottom line: on gpt2, once every measurement artifact is removed — surface leakage, uncorrected layer search, and above all the circular readout — **the temporal belief-state is not demonstrated and its causal claims are refuted.** The one surviving positive is a small belief>null representational separation (p=0.0056, d=0.53). The pipeline, the controls, and the honest readout are all now correct, so this is a real scientific result, not a bug: the paper must be reframed around it (report the negative on gpt2, and test whether a larger instruct model behaves differently before making any positive causal claim).
