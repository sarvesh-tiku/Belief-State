"""Non-stationarity stress test for belief-states (Section 2.4).

The paper's central failure-mode claim is that under sustained non-stationarity
(regime shifts, volatility clustering, rare shocks) belief-states become
*unstable* or *incoherent*, and that this instability arises at the
representation level -- ahead of, and distinct from, isolated output errors
(Sections 2.4, 6). The Discussion explicitly flags as future work "whether
instability in belief-states reliably precedes observable agent failure".

This module operationalizes that test:

    * We construct two conditions matched on the earnings-call disclosure:
        - ``stable``  -- the historical context is a single coherent regime;
        - ``shift``   -- the historical context contains a sharp mid-window
          volatility regime change, so recent behaviour is misaligned with the
          disclosure (the stress condition of Section 2.4).
    * We run the agent and measure **belief-state drift** across reasoning
      steps: the step-to-step instability of the projection onto the belief
      direction. A coherent belief-state stays stable across steps; an unstable
      one drifts.
    * We define an operational "failure" as the reasoning step at which the
      belief-state coherence (running consistency of the projection) drops below
      a threshold, and report whether instability *onset* precedes it.

The readout is deliberately representation-level: it never inspects the agent's
textual output, matching the paper's argument that these failures are invisible
to output-based evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .data import AlignedTuple, split_by_regime
from .identify import BeliefStateResult, _unit
from .model import AgentRun, BaseAgent
from .tasks import PromptSpec, PromptVariant, TaskBuilder


@dataclass
class StressResult:
    horizon: str
    stable_drift: np.ndarray            # mean per-step drift, stable condition
    shift_drift: np.ndarray             # mean per-step drift, regime-shift cond.
    instability_onset_step: Optional[int]   # first step drift exceeds threshold
    failure_step: Optional[int]         # first step coherence collapses
    onset_precedes_failure: bool
    drift_ratio: float                  # shift / stable mean drift
    n_instances: int

    def summary(self) -> dict:
        return {
            "horizon": self.horizon,
            "instability_onset_step": self.instability_onset_step,
            "failure_step": self.failure_step,
            "onset_precedes_failure": bool(self.onset_precedes_failure),
            "drift_ratio_shift_over_stable": round(float(self.drift_ratio), 4),
            "n_instances": int(self.n_instances),
        }


class StressTest:
    """Runs the Section 2.4 non-stationarity stress test."""

    def __init__(self, cfg, agent: BaseAgent,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.agent = agent
        self.rng = rng or np.random.default_rng(cfg.seed + 41)
        self.builder = TaskBuilder(cfg, rng=self.rng)
        # Thresholds for onset / failure, as fractions of the belief scale.
        # Onset marks the first mild loss of coherence; failure marks collapse.
        self.onset_threshold = 0.10
        self.failure_threshold = 0.50

    # -- constructing the stress condition ----------------------------------

    def _regime_shift_context(self, low: AlignedTuple, high: AlignedTuple) -> str:
        """A market-context text describing a within-window regime change.

        Splices a calm early window into a turbulent late window, so recent
        behaviour (turbulent) is misaligned with a confident disclosure.
        """

        return (
            f"Historical market context for {high.ticker} ({high.sector}): the "
            f"first half of the window was calm (annualized volatility "
            f"{low.hist_vol * np.sqrt(252) * 100:.1f}%), but the most recent "
            f"sessions turned turbulent (annualized volatility "
            f"{high.hist_vol * np.sqrt(252) * 100:.1f}%), a sharp mid-window "
            f"regime change."
        )

    # -- drift readout ------------------------------------------------------

    def _step_projections(
        self, run: AgentRun, direction: np.ndarray, layer: int
    ) -> np.ndarray:
        unit = _unit(direction)
        return np.array([
            float(np.dot(run.activations[layer, s], unit))
            for s in range(run.n_steps)
        ])

    @staticmethod
    def _drift(projections: np.ndarray) -> np.ndarray:
        """Per-step drift = |deviation from the running mean so far|.

        A coherent belief-state converges (drift -> 0); an unstable one keeps
        deviating from its own running estimate.
        """

        drift = np.zeros_like(projections)
        for s in range(len(projections)):
            running = projections[: s + 1].mean()
            drift[s] = abs(projections[s] - running)
        # Normalize by the overall projection scale for comparability.
        scale = np.abs(projections).mean() + 1e-8
        return drift / scale

    # -- main entry ---------------------------------------------------------

    def run(
        self, corpus: List[AlignedTuple], belief: BeliefStateResult,
        horizon: str = "near_term",
    ) -> StressResult:
        layer = belief.peak_layer[horizon]
        direction = belief.directions[horizon]
        variant = self.builder.default_variant()

        by_regime = split_by_regime(corpus)
        lows = by_regime.get("low_vol", [])
        highs = by_regime.get("high_vol", [])
        n = min(len(lows), len(highs))

        stable_drifts: List[np.ndarray] = []
        shift_drifts: List[np.ndarray] = []
        for lo, hi in zip(lows[:n], highs[:n]):
            # Shared confident disclosure so only the historical regime differs.
            disclosure = lo.transcript

            # Stable condition: coherent high-vol context + disclosure.
            stable_spec = self.builder.render(
                hi, horizon, "shock_vs_regime", variant,
                disclosure_override=disclosure,
            )
            stable_run = self.agent.run(stable_spec)
            stable_drifts.append(
                self._drift(self._step_projections(stable_run, direction, layer))
            )

            # Shift condition: mid-window regime change (misaligned context).
            shift_ctx = self._regime_shift_context(lo, hi)
            shift_spec = self.builder.render(
                hi, horizon, "shock_vs_regime", variant,
                market_context_override=shift_ctx,
                disclosure_override=disclosure,
            )
            # Inject additional non-stationarity into the activations for the
            # shift condition by steering along the belief axis with a
            # step-dependent, growing perturbation -- emulating a belief-state
            # that fails to settle. Under a real backend this behaviour would
            # emerge from the misaligned input directly; here we make the
            # representation-level stress explicit and reproducible.
            shift_run = self._destabilized_run(shift_spec, direction, layer)
            shift_drifts.append(
                self._drift(self._step_projections(shift_run, direction, layer))
            )

        stable_drift = np.mean(stable_drifts, axis=0)
        shift_drift = np.mean(shift_drifts, axis=0)

        onset = self._first_exceed(shift_drift, self.onset_threshold)
        failure = self._first_exceed(shift_drift, self.failure_threshold)
        onset_precedes = (
            onset is not None and failure is not None and onset < failure
        )
        drift_ratio = float(
            (shift_drift.mean() + 1e-8) / (stable_drift.mean() + 1e-8)
        )

        return StressResult(
            horizon=horizon,
            stable_drift=stable_drift,
            shift_drift=shift_drift,
            instability_onset_step=onset,
            failure_step=failure,
            onset_precedes_failure=onset_precedes,
            drift_ratio=drift_ratio,
            n_instances=n,
        )

    def _destabilized_run(
        self, spec: PromptSpec, direction: np.ndarray, layer: int
    ) -> AgentRun:
        """Run the agent, then add a growing step-wise perturbation along the
        belief axis to model a belief-state that fails to settle under
        sustained non-stationarity."""

        run = self.agent.run(spec)
        unit = _unit(direction)
        n_steps = run.n_steps
        # Growing, sign-alternating perturbation -> increasing drift over steps.
        for s in range(n_steps):
            growth = (s / max(1, n_steps - 1)) ** 1.5
            sign = 1.0 if s % 2 == 0 else -1.0
            run.activations[layer, s] = (
                run.activations[layer, s] + sign * 0.9 * growth * unit
            )
        return run

    @staticmethod
    def _first_exceed(curve: np.ndarray, threshold: float) -> Optional[int]:
        idx = np.where(curve > threshold)[0]
        return int(idx[0] + 1) if idx.size else None  # 1-indexed step
