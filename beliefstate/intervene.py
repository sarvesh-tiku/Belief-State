"""Causal interventions on belief-states (Section 5).

Having identified a candidate belief-state direction ``b`` and the layer of
maximal localization, we test whether it *causally* governs behaviour.  We
apply the steering intervention (Equation 1)

    h' = h + alpha * b

at the peak layer, across the reasoning-step token positions, for a grid of
strengths ``alpha`` (positive and negative, to test reversibility and
directional specificity).  As a control we repeat with matched random
directions ``r`` normalized to the same magnitude (Section 5.1).

Because the paper's downstream signal is a *horizon-consistency score* -- how
coherently the agent maintains expectations across horizons -- we read it off as
the projection of the steered reasoning-step activations onto the belief axis,
relative to the unsteered run.  Under the mock backend this is exactly linear in
alpha for ``b`` and (in expectation) flat for random ``r``, reproducing Figure 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import AlignedTuple
from .identify import BeliefStateResult, _unit
from .model import AgentRun, BaseAgent
from .tasks import PromptSpec, TaskBuilder


@dataclass
class InterventionResult:
    alpha_grid: np.ndarray
    belief_response: np.ndarray          # mean *behavioural* response vs alpha (b)
    belief_response_sem: np.ndarray      # std error across instances
    random_response: np.ndarray          # mean vs alpha (random r), Fig 2A
    random_response_sem: np.ndarray
    # Reversibility / specificity bars (Fig 2B): response at -alpha* and +alpha*.
    reversibility: Dict[str, float]
    directional_specificity: float       # |resp(+a) - resp(-a)| for b vs r
    horizon: str
    # Mechanistic (belief-axis projection) response, kept as a *diagnostic* that
    # is explicitly NOT the causal outcome: it re-reads the intervention axis, so
    # it is expected to move ~linearly with alpha by construction. Reported
    # alongside the behavioural response to show the two are not the same thing.
    belief_projection_response: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Additional specificity controls (behavioural readout):
    #   orthogonal -- steering along directions with b projected out;
    #   swap       -- steering along the *other* horizon's belief direction.
    orthogonal_response: np.ndarray = field(default_factory=lambda: np.empty(0))
    swap_response: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Per-instance behavioural responses at the extreme +alpha (belief vs
    # controls), used for significance testing of the causal effect.
    belief_effect_samples: np.ndarray = field(default_factory=lambda: np.empty(0))
    random_effect_samples: np.ndarray = field(default_factory=lambda: np.empty(0))
    orthogonal_effect_samples: np.ndarray = field(default_factory=lambda: np.empty(0))
    swap_effect_samples: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Firm id per sampled prompt, so the causal-effect test resamples at the
    # firm level (prompts drawn from one firm are correlated).
    effect_clusters: np.ndarray = field(default_factory=lambda: np.empty(0))
    readout: str = "behavioral"
    significance: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "horizon": self.horizon,
            "readout": self.readout,
            "directional_specificity": round(self.directional_specificity, 4),
            "reversibility": {k: round(v, 4) for k, v in self.reversibility.items()},
        }
        if self.significance:
            out["significance"] = self.significance
        return out


class Interventionist:
    """Runs the steering sweep with random-direction controls."""

    def __init__(self, cfg, agent: BaseAgent,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.agent = agent
        self.rng = rng or np.random.default_rng(cfg.seed + 23)
        self.builder = TaskBuilder(cfg, rng=self.rng)

    # -- readouts -----------------------------------------------------------

    def _behavior(self, run: AgentRun, baseline: AgentRun) -> float:
        """Change in the *independent behavioural readout* vs the unsteered run.

        This is the causal outcome: the agent's behavioural channel (output-side
        logit contrast on HF; a distinct downstream axis on the mock), never the
        belief projection. Centered on the baseline so alpha = 0 maps to ~0.
        """

        return self.agent.behavioral_readout(run) - self.agent.behavioral_readout(
            baseline
        )

    def _belief_projection_change(
        self, run: AgentRun, baseline: AgentRun, direction: np.ndarray, layer: int
    ) -> float:
        """Signed change in belief-axis projection at the intervened layer.

        Mechanistic *diagnostic only* -- it re-reads the intervention axis, so it
        moves with alpha by construction. Reported to contrast against the
        behavioural response, not used as the causal outcome.
        """

        unit = _unit(direction)
        steered = run.activations[layer].mean(axis=0)
        base = baseline.activations[layer].mean(axis=0)
        return float(np.dot(steered - base, unit))

    def _matched_random_directions(self, dim: int, n: int) -> List[np.ndarray]:
        dirs = []
        for _ in range(n):
            v = self.rng.standard_normal(dim)
            dirs.append(_unit(v))
        return dirs

    def _orthogonal_directions(
        self, dim: int, n: int, belief: np.ndarray
    ) -> List[np.ndarray]:
        """Random directions with the belief component projected out.

        A steer along these changes the residual with the *same norm* as the
        belief steer but carries no belief-axis content, so any behavioural shift
        they produce is a lower bound on non-specific perturbation effects.
        """

        b = _unit(belief)
        dirs = []
        for _ in range(n):
            v = self.rng.standard_normal(dim)
            v = v - (v @ b) * b
            dirs.append(_unit(v))
        return dirs

    def _sample_prompts(
        self, corpus: List[AlignedTuple], horizon: str
    ) -> Tuple[List[PromptSpec], List[int]]:
        """Sampled prompts plus the firm id backing each, so the causal-effect
        test can resample at the firm level rather than per prompt."""

        variant = self.builder.default_variant()
        n = int(self.cfg.intervene.n_task_instances)
        prompts: List[PromptSpec] = []
        clusters: List[int] = []
        i = 0
        while len(prompts) < n and corpus:
            rec = corpus[i % len(corpus)]
            kind = ("downside_risk", "shock_vs_regime", "scenario_compare")[i % 3]
            prompts.append(self.builder.render(rec, horizon, kind, variant))
            clusters.append(int(rec.firm_id))
            i += 1
        return prompts, clusters

    # -- main sweep ---------------------------------------------------------

    def run(
        self, corpus: List[AlignedTuple], belief: BeliefStateResult,
        horizon: str = "near_term",
    ) -> InterventionResult:
        alpha_grid = np.asarray(self.cfg.intervene.alpha_grid, dtype=float)
        layer = belief.peak_layer[horizon]
        direction = belief.directions[horizon]
        prompts, prompt_clusters = self._sample_prompts(corpus, horizon)
        randoms = self._matched_random_directions(
            self.agent.hidden_dim, int(self.cfg.intervene.n_random_controls)
        )
        orthogonals = self._orthogonal_directions(
            self.agent.hidden_dim, int(self.cfg.intervene.n_random_controls),
            direction,
        )
        # Swap control: the *other* horizon's belief direction (a real, non-random
        # direction that should be less effective if the effect is horizon-specific).
        other_h = [h for h in belief.directions if h != horizon]
        swap_dir = belief.directions[other_h[0]] if other_h else None

        # Baselines (alpha = 0) per prompt, captured once.
        baselines = [self.agent.run(p) for p in prompts]

        n_a, n_p = len(alpha_grid), len(prompts)
        belief_resp = np.zeros((n_a, n_p))       # behavioural (causal outcome)
        belief_proj = np.zeros((n_a, n_p))       # belief-axis projection (diagnostic)
        random_resp = np.zeros((n_a, n_p))
        ortho_resp = np.zeros((n_a, n_p))
        swap_resp = np.zeros((n_a, n_p))

        def mean_behavior(prompt, pi, dirs, alpha):
            vals = []
            for d in dirs:
                run = self.agent.run_with_steering(prompt, layer, d, alpha)
                vals.append(self._behavior(run, baselines[pi]))
            return float(np.mean(vals)) if vals else 0.0

        for ai, alpha in enumerate(alpha_grid):
            for pi, prompt in enumerate(prompts):
                # Belief-direction steering: behavioural outcome + diagnostic proj.
                run_b = self.agent.run_with_steering(prompt, layer, direction, alpha)
                belief_resp[ai, pi] = self._behavior(run_b, baselines[pi])
                belief_proj[ai, pi] = self._belief_projection_change(
                    run_b, baselines[pi], direction, layer
                )
                # Matched random-direction control.
                random_resp[ai, pi] = mean_behavior(prompt, pi, randoms, alpha)
                # Orthogonal-subspace control (b projected out).
                ortho_resp[ai, pi] = mean_behavior(prompt, pi, orthogonals, alpha)
                # Swap control (other horizon's belief direction).
                if swap_dir is not None:
                    run_s = self.agent.run_with_steering(prompt, layer, swap_dir, alpha)
                    swap_resp[ai, pi] = self._behavior(run_s, baselines[pi])

        belief_mean = belief_resp.mean(axis=1)
        belief_sem = belief_resp.std(axis=1) / np.sqrt(max(1, n_p))
        random_mean = random_resp.mean(axis=1)
        random_sem = random_resp.std(axis=1) / np.sqrt(max(1, n_p))

        # Reversibility / specificity at the extreme strengths (Fig 2B).
        a_max_idx = int(np.argmax(alpha_grid))
        a_min_idx = int(np.argmin(alpha_grid))
        reversibility = {
            "belief_neg": float(belief_mean[a_min_idx]),
            "belief_pos": float(belief_mean[a_max_idx]),
            "random_neg": float(random_mean[a_min_idx]),
            "random_pos": float(random_mean[a_max_idx]),
        }
        belief_span = belief_mean[a_max_idx] - belief_mean[a_min_idx]
        random_span = random_mean[a_max_idx] - random_mean[a_min_idx]
        directional_specificity = float(abs(belief_span) - abs(random_span))

        result = InterventionResult(
            alpha_grid=alpha_grid,
            belief_response=belief_mean,
            belief_response_sem=belief_sem,
            random_response=random_mean,
            random_response_sem=random_sem,
            reversibility=reversibility,
            directional_specificity=directional_specificity,
            horizon=horizon,
            belief_projection_response=belief_proj.mean(axis=1),
            orthogonal_response=ortho_resp.mean(axis=1),
            swap_response=swap_resp.mean(axis=1),
            # Per-instance |behavioural response| at maximum steering strength,
            # for the belief-vs-control significance tests of the causal effect.
            belief_effect_samples=np.abs(belief_resp[a_max_idx]),
            random_effect_samples=np.abs(random_resp[a_max_idx]),
            orthogonal_effect_samples=np.abs(ortho_resp[a_max_idx]),
            swap_effect_samples=np.abs(swap_resp[a_max_idx]),
            effect_clusters=np.asarray(prompt_clusters),
        )
        return self._attach_statistics(result)

    def _attach_statistics(
        self, result: InterventionResult, n_permutations: int = 5000
    ) -> InterventionResult:
        """Cluster-aware significance for the causal *behavioural* effect of
        belief steering, against three controls.

        The outcome is the independent behavioural readout (not the belief
        projection), so a significant belief-vs-control gap is genuine causal
        evidence rather than the intervention axis re-read. Steered instances are
        clustered by firm, so each comparison permutes whole firms between arms
        and bootstraps the gap by resampling firms; design-effect-corrected
        effective N is reported. Controls:

          * random     -- matched random directions;
          * orthogonal -- random directions with b projected out (same norm, no
            belief content) -- the strongest specificity control;
          * swap       -- the other horizon's belief direction.
        """

        from .stats import (
            cluster_bootstrap_ci,
            cluster_permutation_test,
            cohens_d,
            effective_sample_size,
        )

        belief = result.belief_effect_samples
        clusters = result.effect_clusters
        if belief.size == 0:
            return result
        if clusters.size != belief.size:
            # Fallback: one cluster per instance (i.e. treat as unclustered).
            clusters = np.arange(belief.size)

        def compare(control: np.ndarray) -> Dict[str, object]:
            test = cluster_permutation_test(
                belief, clusters, control, clusters, self.rng,
                n_permutations=n_permutations, alternative="greater",
            )
            gap = belief - control
            gap_ci = cluster_bootstrap_ci(gap, clusters, self.rng)
            return {
                "permutation": test.as_dict(),
                "effect_size_cohens_d": round(cohens_d(belief, control), 4),
                "gap_ci": gap_ci.as_dict(),
            }

        sig: Dict[str, object] = {
            "outcome": "behavioral_readout (independent of belief axis)",
            "test": "cluster_permutation (firm-level)",
            "effective_n": {
                "nominal": int(belief.size),
                "effective": round(effective_sample_size(belief, clusters), 2),
            },
        }
        if result.random_effect_samples.size:
            sig["belief_vs_random"] = compare(result.random_effect_samples)
        if result.orthogonal_effect_samples.size:
            sig["belief_vs_orthogonal"] = compare(result.orthogonal_effect_samples)
        if result.swap_effect_samples.size and np.any(result.swap_effect_samples):
            sig["belief_vs_swap"] = compare(result.swap_effect_samples)

        # Backwards-compatible top-level keys (run_all + older readers).
        if "belief_vs_random" in sig:
            sig["belief_vs_random_permutation"] = sig["belief_vs_random"]["permutation"]
            sig["effect_size_cohens_d"] = sig["belief_vs_random"]["effect_size_cohens_d"]
        result.significance = sig
        return result
