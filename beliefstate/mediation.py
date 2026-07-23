"""Causal mediation analysis of belief-states (Vig et al., 2020).

The intervention experiment (Section 5) shows that steering the belief
direction ``b`` shifts downstream behaviour. Causal mediation analysis makes
the *mechanism* precise: how much of the effect of a change in future-relevant
input (the treatment: low-vol -> high-vol historical context) on downstream
behaviour is *mediated by* the belief-state, versus flowing through other paths.

Following the neuron/representation mediation framework (Vig et al., 2020) we
decompose the total effect into:

    * Total Effect (TE)            -- behaviour(high-vol) - behaviour(low-vol),
      with everything downstream free to change.
    * Natural Direct Effect (NDE)  -- change the treatment but *hold the
      belief-state fixed* at its control (low-vol) value: the effect that does
      NOT go through the belief-state.
    * Natural Indirect Effect (NIE)-- keep the treatment at control but *set the
      belief-state to its treated (high-vol) value*: the effect carried by the
      belief-state alone.
    * Proportion Mediated          -- NIE / TE.

Two properties make this analysis non-circular, unlike the earlier version the
audit flagged:

  1. **The mediator is patched during the forward pass** (``run_with_patch``),
     not edited on a captured tensor after the fact. The counterfactual
     belief-state actually propagates through the remaining layers, so the
     readout reflects it.
  2. **The readout is independent of the belief axis.** It is the model's
     *behavioural* channel (a next-token risk-vs-calm logit contrast on the HF
     backend; a distinct downstream behaviour axis on the mock), never the
     belief projection re-read. So NIE is not TE by construction: if the belief
     direction does not actually drive behaviour, NIE is ~0 and the proportion
     mediated is small.

The mediator itself is the belief-*direction* component of the peak-layer
residual (rank-1): we substitute only the projection onto ``b`` between the
control and treated runs, leaving the orthogonal complement in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .data import AlignedTuple, split_by_regime
from .identify import BeliefStateResult, _unit
from .model import AgentRun, BaseAgent
from .stats import cluster_bootstrap_ci
from .tasks import TaskBuilder


@dataclass
class MediationResult:
    horizon: str
    total_effect: float
    natural_direct_effect: float
    natural_indirect_effect: float
    proportion_mediated: float
    n_pairs: int
    # Per-pair effects + firm clusters, for a firm-level bootstrap CI on the
    # proportion mediated.
    te_samples: np.ndarray = field(default_factory=lambda: np.empty(0))
    nie_samples: np.ndarray = field(default_factory=lambda: np.empty(0))
    nde_samples: np.ndarray = field(default_factory=lambda: np.empty(0))
    clusters: np.ndarray = field(default_factory=lambda: np.empty(0))
    proportion_mediated_ci: Optional[dict] = None
    readout: str = "behavioral"

    def summary(self) -> dict:
        out = {
            "horizon": self.horizon,
            "readout": self.readout,
            "total_effect": round(float(self.total_effect), 4),
            "natural_direct_effect": round(float(self.natural_direct_effect), 4),
            "natural_indirect_effect": round(float(self.natural_indirect_effect), 4),
            "proportion_mediated": round(float(self.proportion_mediated), 4),
            "n_pairs": int(self.n_pairs),
        }
        if self.proportion_mediated_ci is not None:
            out["proportion_mediated_ci"] = self.proportion_mediated_ci
        return out


class MediationAnalysis:
    """Belief-state mediation via forward-pass patching + independent readout."""

    def __init__(self, cfg, agent: BaseAgent,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.agent = agent
        self.rng = rng or np.random.default_rng(cfg.seed + 53)
        self.builder = TaskBuilder(cfg, rng=self.rng)

    # -- behavioural readout (independent of the belief axis) ---------------

    def _behavior(self, run: AgentRun) -> float:
        """Downstream behavioural response; never the belief projection."""

        return self.agent.behavioral_readout(run)

    @staticmethod
    def _swap_belief_component(
        source: AgentRun, donor: AgentRun, unit: np.ndarray, layer: int
    ) -> np.ndarray:
        """``source``'s peak-layer residual with its belief-axis component
        replaced by ``donor``'s. Returns ``[n_steps, hidden_dim]``.

        This is the rank-1 mediator substitution: only the projection onto
        ``unit`` changes; the orthogonal complement (surface, other features) is
        kept from ``source``.
        """

        h_src = source.activations[layer].copy()          # [S, d]
        h_don = donor.activations[layer]                  # [S, d]
        src_proj = h_src @ unit                           # [S]
        don_proj = h_don @ unit                           # [S]
        return h_src + (don_proj - src_proj)[:, None] * unit[None, :]

    # -- main entry ---------------------------------------------------------

    def run(
        self, corpus: List[AlignedTuple], belief: BeliefStateResult,
        horizon: str = "near_term",
    ) -> MediationResult:
        layer = belief.peak_layer[horizon]
        direction = belief.directions[horizon]
        unit = _unit(direction)
        variant = self.builder.default_variant()

        by_regime = split_by_regime(corpus)
        lows = by_regime.get("low_vol", [])
        highs = by_regime.get("high_vol", [])
        n = min(len(lows), len(highs))

        te_vals, nde_vals, nie_vals, clusters = [], [], [], []
        for lo, hi in zip(lows[:n], highs[:n]):
            disclosure = lo.transcript  # held fixed across treatment
            control_spec = self.builder.render(
                lo, horizon, "downside_risk", variant,
                disclosure_override=disclosure,
            )
            treated_spec = self.builder.render(
                hi, horizon, "downside_risk", variant,
                disclosure_override=disclosure,
            )
            control_run = self.agent.run(control_spec)
            treated_run = self.agent.run(treated_spec)

            beh_control = self._behavior(control_run)
            beh_treated = self._behavior(treated_run)

            # Total effect: treatment on -> off, everything downstream free.
            te = beh_treated - beh_control

            # NDE: treated input, but belief component patched to its control
            # value -- the effect that does NOT flow through the belief axis.
            nde_target = self._swap_belief_component(
                treated_run, control_run, unit, layer
            )
            nde_run = self.agent.run_with_patch(treated_spec, layer, nde_target)
            nde = self._behavior(nde_run) - beh_control

            # NIE: control input, but belief component patched to its treated
            # value -- the effect carried by the belief axis alone.
            nie_target = self._swap_belief_component(
                control_run, treated_run, unit, layer
            )
            nie_run = self.agent.run_with_patch(control_spec, layer, nie_target)
            nie = self._behavior(nie_run) - beh_control

            te_vals.append(te)
            nde_vals.append(nde)
            nie_vals.append(nie)
            clusters.append(int(lo.firm_id))

        te_arr = np.asarray(te_vals)
        nie_arr = np.asarray(nie_vals)
        nde_arr = np.asarray(nde_vals)
        clusters_arr = np.asarray(clusters)

        te = float(te_arr.mean()) if te_arr.size else 0.0
        nde = float(nde_arr.mean()) if nde_arr.size else 0.0
        nie = float(nie_arr.mean()) if nie_arr.size else 0.0
        prop = float(nie / te) if abs(te) > 1e-8 else float("nan")

        # Firm-level bootstrap CI on the proportion mediated (ratio of means),
        # so the reader sees the uncertainty rather than a bare point estimate.
        prop_ci = None
        if te_arr.size >= 2 and abs(te) > 1e-8:
            pairs = np.stack([nie_arr, te_arr], axis=1)  # bootstrap paired rows

            def ratio_of_means(idx_rows: np.ndarray) -> float:
                m = idx_rows.mean(axis=0)
                return float(m[0] / m[1]) if abs(m[1]) > 1e-8 else float("nan")

            ci = cluster_bootstrap_ci(
                pairs, clusters_arr, self.rng,
                statistic=lambda rows: ratio_of_means(np.atleast_2d(rows)),
            )
            prop_ci = ci.as_dict()

        return MediationResult(
            horizon=horizon,
            total_effect=te,
            natural_direct_effect=nde,
            natural_indirect_effect=nie,
            proportion_mediated=prop,
            n_pairs=n,
            te_samples=te_arr,
            nie_samples=nie_arr,
            nde_samples=nde_arr,
            clusters=clusters_arr,
            proportion_mediated_ci=prop_ci,
        )
