"""Mechanistic identification of temporal belief-states (Section 4).

A candidate belief-state is a low-dimensional direction in activation space that
satisfies three *necessary* conditions (Section 4.1):

    1. contrastive sensitivity   -- responds to controlled input changes that
       alter future expectations, while staying insensitive to null (surface)
       perturbations;
    2. temporal persistence      -- persists across intermediate reasoning
       steps rather than only at input/output boundaries;
    3. horizon differentiation   -- distinct directions are required for
       near- vs medium-term reasoning.

Plus a robustness requirement (Section 4.2): a retained direction must remain
detectable and *aligned* across prompt variants; directions that appear only
under a single phrasing are discarded as prompt-specific artifacts.

The pipeline consumes controlled and null contrast pairs (produced by
``tasks.TaskBuilder``), captures activations via a ``model`` backend, and returns
a :class:`BeliefStateResult` carrying the identified directions, the peak layer,
and all the diagnostics needed to reproduce Figure 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import AlignedTuple, split_by_regime
from .model import AgentRun, BaseAgent
from .stats import (
    cluster_bootstrap_ci,
    cluster_permutation_test,
    cohens_d,
    effective_sample_size,
    holm_bonferroni,
)
from .tasks import HORIZONS, PromptSpec, PromptVariant, TaskBuilder


# ---------------------------------------------------------------------------
# Small linear-algebra helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _abs_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.dot(_unit(a), _unit(b))))


def contrast_direction(runs_a: List[AgentRun], runs_b: List[AgentRun], layer: int
                       ) -> np.ndarray:
    """Difference-of-means contrast direction at a layer, averaged over steps.

    This is the primary belief-direction estimator: the mean activation shift
    induced by the controlled contrast, aggregated across reasoning-step tokens.
    """

    def layer_mean(runs: List[AgentRun]) -> np.ndarray:
        mats = [r.activations[layer].mean(axis=0) for r in runs]  # each [d]
        return np.mean(mats, axis=0)

    return _unit(layer_mean(runs_b) - layer_mean(runs_a))


def top_contrast_directions(
    diffs: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k directions maximizing contrastive separation via PCA on diffs.

    ``diffs`` is ``[n_pairs, d]`` of per-pair activation differences.  We return
    the top-k principal directions (columns) and their explained-variance
    ratios -- the low-dimensional belief subspace (Section 4.1).
    """

    if diffs.shape[0] < 2:
        d = diffs.shape[1]
        v = _unit(diffs.mean(axis=0)) if diffs.size else np.zeros(d)
        return v[:, None], np.array([1.0])
    x = diffs - diffs.mean(axis=0, keepdims=True)
    # Economy SVD; right singular vectors are the principal directions.
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(k, vt.shape[0])
    dirs = vt[:k].T  # [d, k]
    var = (s[:k] ** 2)
    ratio = var / (np.sum(s**2) + 1e-12)
    return dirs, ratio


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BeliefStateResult:
    # Per-horizon unit belief direction at the peak layer.
    directions: Dict[str, np.ndarray]
    peak_layer: Dict[str, int]
    # Diagnostics for figures / reporting.
    layerwise_sensitivity: Dict[str, np.ndarray]        # Fig 1A (belief curves)
    null_layerwise_sensitivity: np.ndarray              # Fig 1A (null curve)
    persistence_belief: np.ndarray                      # Fig 1C (belief)
    persistence_null: np.ndarray                        # Fig 1C (null)
    horizon_embedding: Dict[str, np.ndarray]            # Fig 1B (PC1/PC2 per horizon)
    horizon_separation: float                           # cross-horizon distinctness
    cross_variant_alignment: Dict[str, float]           # robustness
    null_sensitivity: float                             # negative control scalar
    conditions: Dict[str, bool]                         # the 3 conditions (+ robustness)
    passed: bool
    # Per-pair projection magnitudes onto the belief direction, used for
    # significance testing (belief contrasts vs null contrasts).
    belief_projections: np.ndarray = field(default_factory=lambda: np.empty(0))
    null_projections: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Firm-level cluster ids aligned with the projections above, so significance
    # testing resamples at the firm level (within-firm pairs are correlated).
    belief_clusters: np.ndarray = field(default_factory=lambda: np.empty(0))
    null_clusters: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Per-horizon layerwise permutation p-values (belief vs null at each layer),
    # for the multiple-comparison-corrected localization.
    layer_pvalues: Dict[str, np.ndarray] = field(default_factory=dict)
    # Populated by attach_statistics() (permutation test + effect size).
    significance: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "peak_layer": self.peak_layer,
            "horizon_separation": round(self.horizon_separation, 4),
            "cross_variant_alignment": {
                k: round(v, 4) for k, v in self.cross_variant_alignment.items()
            },
            "null_sensitivity": round(self.null_sensitivity, 4),
            "conditions": self.conditions,
            "passed": self.passed,
        }
        if self.significance:
            out["significance"] = self.significance
        return out


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------

class BeliefStateIdentifier:
    """Runs the full Section 4 identification + robustness + negative controls."""

    def __init__(self, cfg, agent: BaseAgent,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.agent = agent
        self.rng = rng or np.random.default_rng(cfg.seed + 11)
        self.builder = TaskBuilder(cfg, rng=self.rng)

    # -- contrast-pair construction ----------------------------------------

    def _pair_records(self, corpus: List[AlignedTuple]
                      ) -> Tuple[List[AlignedTuple], List[AlignedTuple]]:
        """Match low- and high-vol records for history contrasts."""

        by_regime = split_by_regime(corpus)
        low = by_regime.get("low_vol", [])
        high = by_regime.get("high_vol", [])
        n = min(len(low), len(high))
        return low[:n], high[:n]

    def _build_contrast_runs(
        self, corpus: List[AlignedTuple], horizon: str, variant: PromptVariant
    ) -> Tuple[List[AgentRun], List[AgentRun], List[int]]:
        """Controlled contrast: vary historical regime, hold disclosure fixed.

        Also returns a cluster id per pair (the low record's firm), so that
        significance testing can resample at the firm level rather than treating
        the two-kinds-per-firm pairs as independent.
        """

        low, high = self._pair_records(corpus)
        runs_a: List[AgentRun] = []
        runs_b: List[AgentRun] = []
        clusters: List[int] = []
        for lo, hi in zip(low, high):
            for kind in ("downside_risk", "shock_vs_regime"):
                pa, pb = self.builder.history_contrast(lo, hi, horizon, kind, variant)
                runs_a.append(self.agent.run(pa))
                runs_b.append(self.agent.run(pb))
                clusters.append(int(lo.firm_id))
        return runs_a, runs_b, clusters

    def _build_null_runs(
        self, corpus: List[AlignedTuple], horizon: str, variant: PromptVariant
    ) -> Tuple[List[AgentRun], List[AgentRun], List[int]]:
        """Null contrasts: reorder / paraphrase / reformat (surface only)."""

        runs_a: List[AgentRun] = []
        runs_b: List[AgentRun] = []
        clusters: List[int] = []
        for rec in corpus:
            for maker in (
                self.builder.null_contrast_reorder,
                self.builder.null_contrast_paraphrase,
                self.builder.null_contrast_format,
            ):
                pa, pb = maker(rec, horizon, "downside_risk", variant)
                runs_a.append(self.agent.run(pa))
                runs_b.append(self.agent.run(pb))
                clusters.append(int(rec.firm_id))
        return runs_a, runs_b, clusters

    # -- layerwise sensitivity (Fig 1A) ------------------------------------

    @staticmethod
    def _layerwise_sensitivity(
        runs_a: List[AgentRun], runs_b: List[AgentRun]
    ) -> np.ndarray:
        """Per-layer contrastive sensitivity = ||mean diff|| aggregated over steps."""

        n_layers = runs_a[0].n_layers
        prof = np.zeros(n_layers)
        for L in range(n_layers):
            a = np.mean([r.activations[L].mean(axis=0) for r in runs_a], axis=0)
            b = np.mean([r.activations[L].mean(axis=0) for r in runs_b], axis=0)
            prof[L] = np.linalg.norm(b - a)
        return prof

    # -- temporal persistence (Fig 1C) -------------------------------------

    def _persistence(
        self, runs_a: List[AgentRun], runs_b: List[AgentRun],
        layer: int, direction: np.ndarray
    ) -> np.ndarray:
        """Projection of the per-step contrast onto the belief direction.

        A high, slowly-decaying curve across steps indicates the representation
        persists through intermediate reasoning rather than at boundaries.
        """

        unit = _unit(direction)
        n_steps = runs_a[0].n_steps
        curve = np.zeros(n_steps)
        for s in range(n_steps):
            a = np.mean([r.activations[layer, s] for r in runs_a], axis=0)
            b = np.mean([r.activations[layer, s] for r in runs_b], axis=0)
            curve[s] = abs(np.dot(b - a, unit))
        # Raw (un-normalized) projection magnitude; the caller normalizes belief
        # and null curves by a *shared* scale so their relative heights are
        # preserved (Figure 1C: the null control sits well below the belief
        # curve rather than being rescaled to it).
        return curve

    @staticmethod
    def _persistence_score(curve: np.ndarray) -> float:
        """Ratio of late-step to early-step magnitude (>~0.6 == persistent)."""

        early = curve[: max(1, len(curve) // 3)].mean()
        late = curve[-max(1, len(curve) // 3):].mean()
        return float(late / (early + 1e-12))

    # -- horizon differentiation (Fig 1B) ----------------------------------

    def _horizon_embedding(
        self, per_horizon_diffs: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """2-D PCA embedding of per-pair diffs, plus a separation score.

        Separation = normalized distance between horizon centroids in the joint
        diff space (silhouette-style).  Partial-but-reproducible separation is
        expected (Section 4.1).
        """

        horizons = list(per_horizon_diffs.keys())
        stacked = np.vstack([per_horizon_diffs[h] for h in horizons])
        labels = np.concatenate(
            [np.full(len(per_horizon_diffs[h]), i) for i, h in enumerate(horizons)]
        )
        mean = stacked.mean(axis=0, keepdims=True)
        xc = stacked - mean
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        pcs = xc @ vt[:2].T  # [N, 2]

        emb: Dict[str, np.ndarray] = {}
        centroids = []
        idx = 0
        for i, h in enumerate(horizons):
            n = len(per_horizon_diffs[h])
            emb[h] = pcs[idx: idx + n]
            centroids.append(emb[h].mean(axis=0))
            idx += n

        # Separation: between-centroid distance normalized by within spread.
        within = np.mean([
            np.mean(np.linalg.norm(emb[h] - emb[h].mean(axis=0), axis=1))
            for h in horizons
        ])
        between = np.linalg.norm(centroids[0] - centroids[1])
        separation = float(between / (between + within + 1e-12))
        return emb, separation

    # -- main entry point ---------------------------------------------------

    def identify(self, corpus: List[AlignedTuple]) -> BeliefStateResult:
        cfg_id = self.cfg.identify
        default_variant = self.builder.default_variant()

        directions: Dict[str, np.ndarray] = {}
        peak_layer: Dict[str, int] = {}
        layerwise: Dict[str, np.ndarray] = {}
        per_horizon_diffs: Dict[str, np.ndarray] = {}
        layer_pvalues: Dict[str, np.ndarray] = {}
        persistence_belief = None
        persistence_null = None
        null_layer_profile = None
        null_sens_values: List[float] = []

        belief_clusters = np.empty(0)
        null_clusters = np.empty(0)
        # Null (surface-only) runs are horizon-agnostic; build once and reuse.
        null_a, null_b, null_cluster_ids = self._build_null_runs(
            corpus, HORIZONS[0], default_variant
        )
        null_clusters = np.asarray(null_cluster_ids)

        for horizon in HORIZONS:
            runs_a, runs_b, contrast_clusters = self._build_contrast_runs(
                corpus, horizon, default_variant
            )
            # Layerwise sensitivity + peak layer.
            prof = self._layerwise_sensitivity(runs_a, runs_b)
            layerwise[horizon] = prof
            L = int(np.argmax(prof))
            peak_layer[horizon] = L

            # Belief direction at the peak layer. The controlled contrast shifts
            # every pair along the same future-expectation axis, so the belief
            # direction is the *mean* activation shift (difference-of-means),
            # not the top principal component of the centered residuals.
            pair_diffs = np.stack([
                _unit(rb.activations[L].mean(axis=0) - ra.activations[L].mean(axis=0))
                for ra, rb in zip(runs_a, runs_b)
            ])
            directions[horizon] = contrast_direction(runs_a, runs_b, L)
            # The low-dimensional belief *subspace* (Section 4.1) is retained as
            # a diagnostic: the mean direction plus the leading residual PCs.
            _subspace, _ = top_contrast_directions(pair_diffs, cfg_id.n_directions)
            per_horizon_diffs[horizon] = pair_diffs

            # Persistence at the peak layer (belief + null), computed once.
            curve = self._persistence(runs_a, runs_b, L, directions[horizon])
            if persistence_belief is None:
                persistence_belief = curve
                persistence_null = self._persistence(
                    null_a, null_b, L, directions[horizon]
                )
                null_layer_profile = self._layerwise_sensitivity(null_a, null_b)
                # Null sensitivity along the belief direction, at peak layer.
                nd = contrast_direction(null_a, null_b, L)
                null_sens_values.append(
                    float(np.linalg.norm(
                        np.mean([r.activations[L].mean(0) for r in null_b], 0)
                        - np.mean([r.activations[L].mean(0) for r in null_a], 0)
                    ))
                )
                # Per-pair projection magnitudes onto the belief direction, for
                # the belief-vs-null significance test (Section 4.2).
                unit = _unit(directions[horizon])
                belief_projections = np.array([
                    abs(np.dot(
                        rb.activations[L].mean(0) - ra.activations[L].mean(0), unit
                    ))
                    for ra, rb in zip(runs_a, runs_b)
                ])
                null_projections = np.array([
                    abs(np.dot(
                        nb.activations[L].mean(0) - na_.activations[L].mean(0), unit
                    ))
                    for na_, nb in zip(null_a, null_b)
                ])
                belief_clusters = np.asarray(contrast_clusters)

                # Per-layer belief-vs-null localization p-values for this
                # horizon, computed from the runs already in hand (no extra
                # forward passes). Cluster-permutation at the firm level; the
                # caller applies Holm-Bonferroni across the layer family.
                layer_pvalues[horizon] = self._layerwise_pvalues(
                    runs_a, runs_b, belief_clusters,
                    null_a, null_b, null_clusters, unit,
                )

        # Normalize belief + null persistence curves by a *shared* scale (the
        # belief curve's peak) so Figure 1C preserves their relative heights.
        shared_scale = float(persistence_belief.max()) + 1e-12
        persistence_belief = persistence_belief / shared_scale
        persistence_null = persistence_null / shared_scale

        # Horizon differentiation (Fig 1B) + separation.
        horizon_embedding, separation = self._horizon_embedding(per_horizon_diffs)

        # Robustness: cross-variant alignment of the belief direction (Sec 4.2).
        cross_variant = self._cross_variant_alignment(corpus, directions)

        # Normalize the null sensitivity relative to belief sensitivity so the
        # threshold is scale-free.
        belief_peak_mag = max(layerwise[h][peak_layer[h]] for h in HORIZONS)
        null_sens = float(np.mean(null_sens_values) / (belief_peak_mag + 1e-12))

        # --- evaluate the three necessary conditions + robustness ----------
        contrastive_sensitivity = belief_peak_mag  # absolute belief signal
        persist_score = self._persistence_score(persistence_belief)
        cond = {
            "contrastive_sensitivity": bool(
                contrastive_sensitivity > cfg_id.min_contrastive_sensitivity
                and null_sens < cfg_id.max_null_sensitivity
            ),
            "temporal_persistence": bool(
                persist_score > cfg_id.min_temporal_persistence
            ),
            "horizon_differentiation": bool(
                separation > cfg_id.min_horizon_separation
            ),
            "cross_variant_robustness": bool(
                min(cross_variant.values()) > cfg_id.min_cross_variant_alignment
            ),
        }
        passed = all(cond.values())

        return BeliefStateResult(
            directions=directions,
            peak_layer=peak_layer,
            layerwise_sensitivity=layerwise,
            null_layerwise_sensitivity=null_layer_profile,
            persistence_belief=persistence_belief,
            persistence_null=persistence_null,
            horizon_embedding=horizon_embedding,
            horizon_separation=separation,
            cross_variant_alignment=cross_variant,
            null_sensitivity=null_sens,
            conditions=cond,
            passed=passed,
            belief_projections=belief_projections,
            null_projections=null_projections,
            belief_clusters=belief_clusters,
            null_clusters=null_clusters,
            layer_pvalues=layer_pvalues,
        )

    # -- per-layer localization p-values (multiple-comparison family) -------

    def _layerwise_pvalues(
        self,
        runs_a: List[AgentRun],
        runs_b: List[AgentRun],
        belief_clusters: np.ndarray,
        null_a: List[AgentRun],
        null_b: List[AgentRun],
        null_clusters: np.ndarray,
        unit: np.ndarray,
        n_permutations: int = 2000,
    ) -> np.ndarray:
        """Per-layer belief-vs-null separation p-values.

        For each layer we project the per-pair contrast (belief) and the
        per-pair null contrast onto the *peak-layer* belief direction ``unit``,
        then run a firm-clustered permutation test of belief > null at that
        layer. This localizes where the belief signal separates from surface
        perturbations. The caller applies Holm-Bonferroni across this array so
        the layer-by-layer search does not inflate the family-wise error rate.

        Uses the same ``unit`` axis at every layer (rather than re-deriving a
        direction per layer) so the profile is comparable across depth and does
        not manufacture a per-layer axis that trivially separates its own data.
        """

        n_layers = runs_a[0].n_layers
        pvals = np.ones(n_layers)
        for L in range(n_layers):
            belief_proj = np.array([
                abs(np.dot(
                    rb.activations[L].mean(0) - ra.activations[L].mean(0), unit
                ))
                for ra, rb in zip(runs_a, runs_b)
            ])
            null_proj = np.array([
                abs(np.dot(
                    nb.activations[L].mean(0) - na_.activations[L].mean(0), unit
                ))
                for na_, nb in zip(null_a, null_b)
            ])
            test = cluster_permutation_test(
                belief_proj, belief_clusters, null_proj, null_clusters,
                self.rng, n_permutations=n_permutations, alternative="greater",
            )
            pvals[L] = test.p_value
        return pvals

    # -- significance testing (belief vs null contrasts) --------------------

    def attach_statistics(
        self, result: BeliefStateResult, n_permutations: int = 5000
    ) -> BeliefStateResult:
        """Attach cluster-aware significance for the belief-vs-null comparison.

        The experimental units are clustered by firm (two contrast kinds and
        three null makers share a firm), so an i.i.d. test over-counts evidence.
        We use a firm-clustered permutation test, firm-cluster bootstrap CIs, and
        report the design-effect-corrected effective N so the reader can see how
        much the clustering shrinks the nominal sample. Holm-Bonferroni adjusts
        the per-layer localization p-values across the layer family.
        """

        belief = result.belief_projections
        null = result.null_projections
        if belief.size == 0 or null.size == 0:
            return result

        test = cluster_permutation_test(
            belief, result.belief_clusters, null, result.null_clusters,
            self.rng, n_permutations=n_permutations, alternative="greater",
        )
        ci = cluster_bootstrap_ci(belief, result.belief_clusters, self.rng)
        null_ci = cluster_bootstrap_ci(null, result.null_clusters, self.rng)
        n_eff_belief = effective_sample_size(belief, result.belief_clusters)
        n_eff_null = effective_sample_size(null, result.null_clusters)

        significance: Dict[str, object] = {
            "belief_vs_null_permutation": test.as_dict(),
            "test": "cluster_permutation (firm-level)",
            "effect_size_cohens_d": round(cohens_d(belief, null), 4),
            "belief_projection_ci": ci.as_dict(),
            "null_projection_ci": null_ci.as_dict(),
            "effective_n": {
                "belief_nominal": int(belief.size),
                "belief_effective": round(n_eff_belief, 2),
                "null_nominal": int(null.size),
                "null_effective": round(n_eff_null, 2),
            },
        }

        # Multiple-comparison correction across the per-layer localization
        # search, per horizon and across the pooled layer family.
        if result.layer_pvalues:
            per_horizon = {}
            for horizon, pv in result.layer_pvalues.items():
                pv = np.asarray(pv, float)
                corrected = holm_bonferroni(pv, alpha=0.05)
                sig_layers = [int(i) for i, r in enumerate(corrected["reject"]) if r]
                per_horizon[horizon] = {
                    "significant_layers": sig_layers,
                    "n_significant": len(sig_layers),
                    "min_p_adjusted": round(float(min(corrected["p_adjusted"])), 5),
                }
            significance["layer_localization"] = {
                "correction": "holm_bonferroni",
                "per_horizon": per_horizon,
            }

        result.significance = significance
        return result

    # -- robustness across prompt variants (Section 4.2) --------------------

    def _cross_variant_alignment(
        self, corpus: List[AlignedTuple], reference: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Mean |cosine| between the reference direction and per-variant
        re-identified directions, per horizon."""

        out: Dict[str, float] = {}
        variants = self.builder.all_variants()
        for horizon in HORIZONS:
            L = None
            aligns: List[float] = []
            for variant in variants:
                runs_a, runs_b, _ = self._build_contrast_runs(corpus, horizon, variant)
                prof = self._layerwise_sensitivity(runs_a, runs_b)
                L = int(np.argmax(prof))
                d = contrast_direction(runs_a, runs_b, L)
                aligns.append(_abs_cosine(d, reference[horizon]))
            out[horizon] = float(np.mean(aligns))
        return out


# ---------------------------------------------------------------------------
# Causal (activation-patching) direction validation
# ---------------------------------------------------------------------------

@dataclass
class PatchingDirectionResult:
    horizon: str
    layer: int
    # Candidate directions scored by the causal behavioural effect of patching a
    # fixed increment along each (independent readout, not the projection).
    candidate_labels: List[str]
    causal_effects: np.ndarray            # mean |Δ behaviour| per candidate
    diff_of_means_effect: float           # the difference-of-means candidate's effect
    best_label: str
    best_effect: float
    # Alignment of the best causal direction with the difference-of-means axis:
    # a high value means the correlational and causal direction finders agree.
    best_vs_diff_alignment: float

    def summary(self) -> dict:
        return {
            "horizon": self.horizon,
            "layer": int(self.layer),
            "candidates": {
                lab: round(float(e), 4)
                for lab, e in zip(self.candidate_labels, self.causal_effects)
            },
            "diff_of_means_effect": round(float(self.diff_of_means_effect), 4),
            "best": self.best_label,
            "best_effect": round(float(self.best_effect), 4),
            "best_vs_diff_alignment": round(float(self.best_vs_diff_alignment), 4),
        }


class PatchingDirectionFinder:
    """Find/validate the belief direction by its *causal* effect, not just its
    correlation with the contrast.

    Difference-of-means asks "which direction moves most when the input changes?"
    -- a correlational question. This finder asks the causal one: "patching a
    fixed step along which direction most changes the model's behaviour?", where
    behaviour is the backend's independent readout (never the patched axis). It
    scores a small candidate set (the difference-of-means axis plus the leading
    PCs of the per-pair contrast) and reports which one the model's own
    computation is most sensitive to, and whether that agrees with
    difference-of-means. Agreement is evidence the correlational direction is
    also the causal one; disagreement is a red flag the paper should report.
    """

    def __init__(self, cfg, agent: BaseAgent,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.agent = agent
        self.rng = rng or np.random.default_rng(cfg.seed + 29)
        self.builder = TaskBuilder(cfg, rng=self.rng)

    def _candidate_directions(
        self, pair_diffs: np.ndarray, k: int
    ) -> Tuple[List[str], List[np.ndarray]]:
        labels = ["diff_of_means"]
        dirs = [_unit(pair_diffs.mean(axis=0))]
        subspace, _ = top_contrast_directions(pair_diffs, k)
        for j in range(subspace.shape[1]):
            labels.append(f"pc{j+1}")
            dirs.append(_unit(subspace[:, j]))
        return labels, dirs

    def run(
        self, corpus: List[AlignedTuple], belief: BeliefStateResult,
        horizon: str = "near_term", alpha: Optional[float] = None,
        n_prompts: int = 8,
    ) -> PatchingDirectionResult:
        layer = belief.peak_layer[horizon]
        variant = self.builder.default_variant()
        if alpha is None:
            grid = np.asarray(self.cfg.intervene.alpha_grid, float)
            alpha = float(np.max(np.abs(grid))) or 1.0

        # Per-pair contrast diffs at the peak layer -> candidate directions.
        by_regime = split_by_regime(corpus)
        lows = by_regime.get("low_vol", [])
        highs = by_regime.get("high_vol", [])
        n = min(len(lows), len(highs))
        pair_diffs = []
        for lo, hi in zip(lows[:n], highs[:n]):
            pa, pb = self.builder.history_contrast(
                lo, hi, horizon, "downside_risk", variant
            )
            ra, rb = self.agent.run(pa), self.agent.run(pb)
            pair_diffs.append(
                rb.activations[layer].mean(0) - ra.activations[layer].mean(0)
            )
        pair_diffs = np.stack(pair_diffs)
        labels, dirs = self._candidate_directions(
            pair_diffs, int(self.cfg.identify.n_directions)
        )

        # A fixed set of prompts to patch.
        prompts = [
            self.builder.render(lows[i % max(1, len(lows))], horizon,
                                "downside_risk", variant)
            for i in range(n_prompts)
        ] if lows else []

        # Causal effect of each candidate: mean |Δ behaviour| under +alpha steer
        # (steering along a unit direction is a rank-1 additive patch), read via
        # the backend's independent behavioural channel.
        effects = np.zeros(len(dirs))
        for di, d in enumerate(dirs):
            deltas = []
            for p in prompts:
                base = self.agent.run(p)
                steered = self.agent.run_with_steering(p, layer, d, alpha)
                deltas.append(abs(
                    self.agent.behavioral_readout(steered)
                    - self.agent.behavioral_readout(base)
                ))
            effects[di] = float(np.mean(deltas)) if deltas else 0.0

        diff_effect = float(effects[0])
        best_idx = int(np.argmax(effects))
        best_align = _abs_cosine(dirs[best_idx], dirs[0])
        return PatchingDirectionResult(
            horizon=horizon,
            layer=layer,
            candidate_labels=labels,
            causal_effects=effects,
            diff_of_means_effect=diff_effect,
            best_label=labels[best_idx],
            best_effect=float(effects[best_idx]),
            best_vs_diff_alignment=best_align,
        )
