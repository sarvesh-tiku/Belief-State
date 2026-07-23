"""Reproduction of the paper's figures.

Figure 1 (identification pipeline):
    1A  Layerwise Sensitivity Profile   -- contrastive sensitivity vs layer,
        for near-term, medium-term, and the null control.
    1B  Horizon Differentiation         -- PC1/PC2 scatter of per-pair contrast
        diffs, colored by horizon (and regime).
    1C  Persistence Across Reasoning Steps -- belief vs null activation magnitude
        across the 8 reasoning steps.

Figure 2 (causal interventions):
    2A  Intervention Strength vs Behavioral Shift -- horizon-consistency vs alpha
        for the belief direction b vs matched random directions r.
    2B  Reversibility and Specificity   -- response at -alpha and +alpha, belief
        vs random.

All figures use matplotlib with a non-interactive backend so they render in
headless runs.
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .identify import BeliefStateResult  # noqa: E402
from .intervene import InterventionResult  # noqa: E402
from .probes import ProbeResult  # noqa: E402
from .stress import StressResult  # noqa: E402

_BLUE = "#1f77b4"
_ORANGE = "#ff7f0e"
_GREEN = "#2ca02c"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def plot_figure1(belief: BeliefStateResult, out_dir: str) -> str:
    _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    # --- 1A: Layerwise Sensitivity Profile ---
    ax = axes[0]
    near = belief.layerwise_sensitivity["near_term"]
    med = belief.layerwise_sensitivity["medium_term"]
    null = belief.null_layerwise_sensitivity
    layers = np.arange(len(near))
    # Normalize each to its own peak for a clean 0..1 comparison.
    ax.plot(layers, near / (near.max() + 1e-12), color=_BLUE, label="near-term")
    ax.plot(layers, med / (med.max() + 1e-12), color=_ORANGE, label="medium-term")
    ax.plot(layers, null / (near.max() + 1e-12), color=_GREEN, label="null control")
    ax.set_title("Figure 1A: Layerwise Sensitivity Profile", fontsize=9)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Contrastive Sensitivity")
    ax.legend(fontsize=6, loc="upper left")

    # --- 1B: Horizon Differentiation in Activation Space ---
    ax = axes[1]
    emb = belief.horizon_embedding
    ax.scatter(emb["near_term"][:, 0], emb["near_term"][:, 1],
               s=14, color=_BLUE, alpha=0.8, label="near-term")
    ax.scatter(emb["medium_term"][:, 0], emb["medium_term"][:, 1],
               s=14, color=_ORANGE, alpha=0.8, label="medium-term")
    ax.set_title("Figure 1B: Horizon Differentiation in Activation Space",
                 fontsize=9)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=6, loc="upper right")

    # --- 1C: Persistence Across Reasoning Steps ---
    ax = axes[2]
    steps = np.arange(1, len(belief.persistence_belief) + 1)
    ax.plot(steps, belief.persistence_belief, color=_BLUE, marker="o",
            markersize=3, label="belief-state")
    ax.plot(steps, belief.persistence_null, color=_ORANGE, marker="s",
            markersize=3, label="null control")
    ax.set_title("Figure 1C: Persistence Across Reasoning Steps", fontsize=9)
    ax.set_xlabel("Reasoning Step")
    ax.set_ylabel("Activation Magnitude")
    ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    path = os.path.join(out_dir, "figure1.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

def plot_figure2(interv: InterventionResult, out_dir: str) -> str:
    _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # --- 2A: Intervention Strength vs Behavioral Shift ---
    ax = axes[0]
    a = interv.alpha_grid
    ax.errorbar(a, interv.belief_response, yerr=interv.belief_response_sem,
                color=_BLUE, marker="o", markersize=3, capsize=2,
                label="belief direction b")
    ax.errorbar(a, interv.random_response, yerr=interv.random_response_sem,
                color=_ORANGE, marker="x", markersize=4, capsize=2,
                label="random directions r")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.axvline(0.0, color="0.6", lw=0.6)
    ax.set_title("Figure 2A: Intervention Strength vs Behavioral Shift",
                 fontsize=9)
    ax.set_xlabel(r"Intervention Strength ($\alpha$)")
    ax.set_ylabel("Horizon-Consistency Score")
    ax.legend(fontsize=6, loc="upper left")

    # --- 2B: Reversibility and Specificity ---
    ax = axes[1]
    rev = interv.reversibility
    positions = [0, 1]  # -alpha, +alpha
    width = 0.35
    belief_vals = [rev["belief_neg"], rev["belief_pos"]]
    random_vals = [rev["random_neg"], rev["random_pos"]]
    ax.bar([p - width / 2 for p in positions], belief_vals, width,
           color=_BLUE, label="belief b")
    ax.bar([p + width / 2 for p in positions], random_vals, width,
           color=_ORANGE, label="random r")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([r"$-\alpha$", r"$+\alpha$"])
    ax.set_title("Figure 2B: Reversibility and Specificity", fontsize=9)
    ax.set_xlabel("Intervention Condition")
    ax.set_ylabel("Horizon-Consistency Score")
    ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "figure2.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: non-stationarity stress test + probe corroboration
# ---------------------------------------------------------------------------

def plot_figure3(
    stress: "StressResult", probes: Dict[str, "ProbeResult"], out_dir: str
) -> str:
    """Figure 3: belief-state instability under regime shift (A) and the
    convergent probe-accuracy localization (B)."""

    _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # --- 3A: belief-state drift across a sustained regime shift ---
    ax = axes[0]
    steps = np.arange(1, len(stress.stable_drift) + 1)
    ax.plot(steps, stress.stable_drift, color=_BLUE, marker="o", markersize=3,
            label="stable regime")
    ax.plot(steps, stress.shift_drift, color=_ORANGE, marker="s", markersize=3,
            label="regime shift")
    if stress.failure_step is not None:
        ax.axvline(stress.failure_step, color="0.5", ls="--", lw=0.8,
                   label="observed failure")
    ax.set_title("Figure 3A: Belief-State Instability Under Regime Shift",
                 fontsize=9)
    ax.set_xlabel("Reasoning Step")
    ax.set_ylabel("Belief-State Drift")
    ax.legend(fontsize=6, loc="upper left")

    # --- 3B: probe-accuracy localization (convergent with Fig 1A) ---
    ax = axes[1]
    for horizon, color in (("near_term", _BLUE), ("medium_term", _ORANGE)):
        pr = probes[horizon]
        layers = np.arange(len(pr.layer_accuracy))
        ax.plot(layers, pr.layer_accuracy, color=color, label=f"{horizon} probe")
        ax.fill_between(
            layers,
            pr.layer_accuracy - pr.layer_accuracy_std,
            pr.layer_accuracy + pr.layer_accuracy_std,
            color=color, alpha=0.15,
        )
    ax.axhline(0.5, color="0.6", lw=0.6, ls=":", label="chance")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Figure 3B: Probe-Decodability of Belief-State by Layer",
                 fontsize=9)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Cross-Validated Probe Accuracy")
    ax.legend(fontsize=6, loc="lower center")

    fig.tight_layout()
    path = os.path.join(out_dir, "figure3.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
