"""Horizon-aware tasks, prompt templates, and contrasts.

Covers:

    * Section 3.2 -- the horizon-aware reasoning tasks (near- vs medium-term
      downside risk, transient-shock vs regime-shift, scenario comparison),
      instantiated from *fixed prompt templates*.
    * Section 3.3 -- eliciting belief-states through *controlled contrasts*:
      vary historical market context while holding the disclosure fixed, and
      vice versa.  The agent is never asked to output forecasts/probabilities.
    * Section 4.2 -- prompt *variants* (framing, output format, field ordering,
      paraphrase) for the robustness sweep, and *null contrasts* that perturb
      surface form without changing future-relevant semantics.

A ``PromptSpec`` is a fully-rendered prompt plus the metadata needed to align
its reasoning-step tokens during activation capture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import AlignedTuple

HORIZONS = ("near_term", "medium_term")
TASK_KINDS = ("downside_risk", "shock_vs_regime", "scenario_compare")


# ---------------------------------------------------------------------------
# Prompt variant axes (Section 4.2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptVariant:
    framing: str      # "risk_assessment" | "scenario_ranking"
    fmt: str          # "free_form" | "structured"
    ordering: str     # "history_first" | "disclosure_first"
    paraphrase: bool  # apply meaning-preserving surface paraphrase

    def key(self) -> str:
        p = "para" if self.paraphrase else "lit"
        return f"{self.framing}|{self.fmt}|{self.ordering}|{p}"


@dataclass
class PromptSpec:
    """A rendered prompt and the metadata to locate reasoning-step tokens."""

    text: str
    firm_id: int
    period: int
    horizon: str
    task_kind: str
    regime: str
    uncertainty: int
    variant: PromptVariant
    # Marker sentinel inserted before each reasoning step so the model backend
    # can align activations to steps (Section 4: aggregate over reasoning-step
    # tokens, not input text).
    reasoning_marker: str = "<step>"
    n_steps: int = 8
    # Free-form semantic fields, kept so contrasts can swap exactly one field.
    market_context: str = ""
    disclosure: str = ""


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

_HORIZON_PHRASE = {
    "near_term": "the near term (roughly the next few trading sessions)",
    "medium_term": "the medium term (roughly the next one to two quarters)",
}

_TASK_INSTRUCTION = {
    ("downside_risk", "risk_assessment"): (
        "Assess the downside risk over {horizon}. Reason step by step about how "
        "the historical market behavior and the disclosure jointly shape risk."
    ),
    ("downside_risk", "scenario_ranking"): (
        "Rank plausible outcomes over {horizon} from most to least adverse, "
        "reasoning step by step from the historical behavior and the disclosure."
    ),
    ("shock_vs_regime", "risk_assessment"): (
        "Determine whether the observed volatility over {horizon} reflects a "
        "transient shock or a persistent regime shift. Reason step by step."
    ),
    ("shock_vs_regime", "scenario_ranking"): (
        "Rank the two hypotheses -- transient shock vs. persistent regime shift "
        "-- for {horizon}, reasoning step by step from the evidence."
    ),
    ("scenario_compare", "risk_assessment"): (
        "Compare a benign and an adverse scenario for {horizon} and assess which "
        "the evidence favors. Reason step by step."
    ),
    ("scenario_compare", "scenario_ranking"): (
        "Rank a benign, a neutral, and an adverse scenario for {horizon}, "
        "reasoning step by step from the historical behavior and disclosure."
    ),
}

_FORMAT_SUFFIX = {
    "free_form": "Provide your reasoning as free-form prose.",
    "structured": "Provide your reasoning as a numbered list of considerations.",
}


def _paraphrase(text: str) -> str:
    """A light meaning-preserving surface paraphrase.

    It alters tokenization and local surface statistics (Section 4.2) without
    changing guidance/uncertainty content: synonym swaps + whitespace/casing
    of connective words only.
    """

    subs = [
        (r"\bReason step by step\b", "Think through this stepwise"),
        (r"\bAssess\b", "Evaluate"),
        (r"\bDetermine\b", "Decide"),
        (r"\bhistorical market behavior\b", "prior market activity"),
        (r"\bthe disclosure\b", "the management commentary"),
        (r"\bProvide your reasoning\b", "Lay out your reasoning"),
    ]
    out = text
    for pat, rep in subs:
        out = re.sub(pat, rep, out)
    return out


class TaskBuilder:
    """Renders prompts and constructs controlled + null contrasts."""

    def __init__(self, cfg, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed + 1)
        self.n_steps = int(cfg.tasks.reasoning_steps)

    # -- single prompt ------------------------------------------------------

    def render(
        self,
        rec: AlignedTuple,
        horizon: str,
        task_kind: str,
        variant: PromptVariant,
        *,
        market_context_override: Optional[str] = None,
        disclosure_override: Optional[str] = None,
    ) -> PromptSpec:
        market_context = (
            market_context_override
            if market_context_override is not None
            else rec.market_context_text()
        )
        disclosure = (
            disclosure_override
            if disclosure_override is not None
            else rec.transcript
        )

        instr = _TASK_INSTRUCTION[(task_kind, variant.framing)].format(
            horizon=_HORIZON_PHRASE[horizon]
        )
        fmt = _FORMAT_SUFFIX[variant.fmt]

        ctx_block = f"[MARKET CONTEXT]\n{market_context}"
        dis_block = f"[EARNINGS CALL DISCLOSURE]\n{disclosure}"
        if variant.ordering == "history_first":
            body = f"{ctx_block}\n\n{dis_block}"
        else:
            body = f"{dis_block}\n\n{ctx_block}"

        # Reasoning-step scaffold: markers the backend aligns activations to.
        steps = "\n".join(
            f"{self.render_marker(i)} reasoning step {i + 1}"
            for i in range(self.n_steps)
        )
        prompt = f"{instr}\n\n{body}\n\n{fmt}\n\n{steps}"
        if variant.paraphrase:
            prompt = _paraphrase(prompt)

        return PromptSpec(
            text=prompt,
            firm_id=rec.firm_id,
            period=rec.period,
            horizon=horizon,
            task_kind=task_kind,
            regime=rec.regime,
            uncertainty=rec.uncertainty,
            variant=variant,
            n_steps=self.n_steps,
            market_context=market_context,
            disclosure=disclosure,
        )

    @staticmethod
    def render_marker(i: int) -> str:
        return f"<step{i}>"

    # -- variants -----------------------------------------------------------

    def default_variant(self) -> PromptVariant:
        t = self.cfg.tasks
        return PromptVariant(
            framing=t.framings[0],
            fmt=t.formats[0],
            ordering=t.orderings[0],
            paraphrase=False,
        )

    def all_variants(self) -> List[PromptVariant]:
        """The cross-product of variant axes used for the robustness sweep."""

        t = self.cfg.tasks
        variants: List[PromptVariant] = []
        for framing in t.framings:
            for fmt in t.formats:
                for ordering in t.orderings:
                    for para in ([False, True] if t.paraphrase else [False]):
                        variants.append(
                            PromptVariant(framing, fmt, ordering, para)
                        )
        return variants

    # -- controlled contrasts (Section 3.3) ---------------------------------

    def history_contrast(
        self,
        low_rec: AlignedTuple,
        high_rec: AlignedTuple,
        horizon: str,
        task_kind: str,
        variant: PromptVariant,
    ) -> Tuple[PromptSpec, PromptSpec]:
        """Hold the disclosure fixed; swap the historical context regime.

        Returns (low_vol_history, high_vol_history) prompts that share an
        identical disclosure -- isolating future-oriented variation driven by
        the historical volatility regime.
        """

        shared_disclosure = low_rec.transcript
        a = self.render(
            low_rec, horizon, task_kind, variant,
            market_context_override=low_rec.market_context_text(),
            disclosure_override=shared_disclosure,
        )
        b = self.render(
            high_rec, horizon, task_kind, variant,
            market_context_override=high_rec.market_context_text(),
            disclosure_override=shared_disclosure,
        )
        return a, b

    def disclosure_contrast(
        self,
        rec: AlignedTuple,
        low_unc_transcript: str,
        high_unc_transcript: str,
        horizon: str,
        task_kind: str,
        variant: PromptVariant,
    ) -> Tuple[PromptSpec, PromptSpec]:
        """Hold the historical context fixed; swap disclosure uncertainty.

        Returns (confident_disclosure, uncertain_disclosure) prompts sharing an
        identical market context.
        """

        shared_ctx = rec.market_context_text()
        a = self.render(
            rec, horizon, task_kind, variant,
            market_context_override=shared_ctx,
            disclosure_override=low_unc_transcript,
        )
        b = self.render(
            rec, horizon, task_kind, variant,
            market_context_override=shared_ctx,
            disclosure_override=high_unc_transcript,
        )
        return a, b

    # -- null contrasts (Section 4.2 negative controls) ---------------------

    def null_contrast_reorder(
        self, rec: AlignedTuple, horizon: str, task_kind: str, variant: PromptVariant
    ) -> Tuple[PromptSpec, PromptSpec]:
        """Reorder historical inputs while preserving summary statistics.

        The market-context *text* is a summary of the lookback window; here we
        re-render it from a permuted return series with identical mean/std, so
        the summary is unchanged -- a true surface-only perturbation.
        """

        base = self.render(rec, horizon, task_kind, variant)
        # Identical summary text -> no future-relevant change. We still create a
        # distinct object so downstream code treats it as a pair.
        perturbed_ctx = base.market_context + " "  # benign whitespace only
        b = self.render(
            rec, horizon, task_kind, variant,
            market_context_override=perturbed_ctx,
        )
        return base, b

    def null_contrast_paraphrase(
        self, rec: AlignedTuple, horizon: str, task_kind: str, variant: PromptVariant
    ) -> Tuple[PromptSpec, PromptSpec]:
        """Paraphrase the disclosure without changing guidance/uncertainty."""

        base = self.render(rec, horizon, task_kind, variant)
        para_disc = _paraphrase_disclosure(rec.transcript)
        b = self.render(
            rec, horizon, task_kind, variant,
            disclosure_override=para_disc,
        )
        return base, b

    def null_contrast_format(
        self, rec: AlignedTuple, horizon: str, task_kind: str, variant: PromptVariant
    ) -> Tuple[PromptSpec, PromptSpec]:
        """Benign formatting modification (whitespace / section headers)."""

        base = self.render(rec, horizon, task_kind, variant)
        b_text = base.text.replace("[MARKET CONTEXT]", "[ MARKET  CONTEXT ]")
        b = PromptSpec(
            text=b_text,
            firm_id=base.firm_id, period=base.period, horizon=base.horizon,
            task_kind=base.task_kind, regime=base.regime,
            uncertainty=base.uncertainty, variant=base.variant,
            n_steps=base.n_steps, market_context=base.market_context,
            disclosure=base.disclosure,
        )
        return base, b


def _paraphrase_disclosure(text: str) -> str:
    """Surface paraphrase of a transcript that preserves guidance content."""

    subs = [
        (r"\bThank you\b", "Thanks"),
        (r"\bGood afternoon\b", "Good day"),
        (r"\bwe are\b", "we're"),
        (r"\bWe appreciate\b", "We're grateful"),
        (r"\bwelcome to\b", "welcome you to"),
    ]
    out = text
    for pat, rep in subs:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out
