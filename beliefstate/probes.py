"""Cross-validated linear probes for temporal belief-states.

Difference-of-means (Section 4) gives one estimate of the belief direction. As
an independent check we *train* linear probes to decode the future-relevant
label (the volatility regime that the historical context was drawn from) from
the reasoning-step activations at each layer, using k-fold cross-validation.

Two things matter for the paper's claims:

    * The **layerwise probe-accuracy profile** should peak where the
      difference-of-means sensitivity peaks (Figure 1A) -- two methods, one
      localization.  A probe that generalizes across folds indicates a linearly
      decodable belief-state, not an artifact of a single contrast.
    * The learned probe **weight vector** should align with the
      difference-of-means belief direction (convergent validity).

Probes are plain regularized logistic-regression models (scikit-learn) trained
on standardized activations, with grouped folds so that reasoning steps from the
same task instance never straddle the train/test split (no leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import AlignedTuple, split_by_regime
from .model import AgentRun, BaseAgent
from .tasks import HORIZONS, PromptVariant, TaskBuilder


@dataclass
class ProbeResult:
    horizon: str
    layer_accuracy: np.ndarray          # mean CV accuracy per layer
    layer_accuracy_std: np.ndarray      # std across folds per layer
    best_layer: int
    best_accuracy: float
    chance: float
    weight_alignment: float             # |cos| between probe weights and DoM dir
    n_samples: int

    def summary(self) -> Dict[str, object]:
        return {
            "horizon": self.horizon,
            "best_layer": int(self.best_layer),
            "best_accuracy": round(float(self.best_accuracy), 4),
            "chance": round(float(self.chance), 4),
            "weight_alignment_with_diff_of_means": round(
                float(self.weight_alignment), 4
            ),
            "n_samples": int(self.n_samples),
        }


def _standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd, mu, sd


class BeliefProbe:
    """Trains + cross-validates linear probes across layers for one horizon."""

    def __init__(self, cfg, agent: BaseAgent,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.agent = agent
        self.rng = rng or np.random.default_rng(cfg.seed + 31)
        self.builder = TaskBuilder(cfg, rng=self.rng)
        self.n_folds = 5

    # -- dataset construction ----------------------------------------------

    def _collect(
        self, corpus: List[AlignedTuple], horizon: str, variant: PromptVariant
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, y, groups).

        X: [N, n_layers, d] mean-over-steps activations per instance.
        y: regime label (0 low-vol, 1 high-vol).
        groups: task-instance id (for grouped CV).
        """

        by_regime = split_by_regime(corpus)
        feats: List[np.ndarray] = []
        labels: List[int] = []
        groups: List[int] = []
        gid = 0
        for label, regime in enumerate(("low_vol", "high_vol")):
            for rec in by_regime.get(regime, []):
                for kind in ("downside_risk", "shock_vs_regime"):
                    spec = self.builder.render(rec, horizon, kind, variant)
                    run = self.agent.run(spec)
                    # mean over reasoning steps -> [n_layers, d]
                    feats.append(run.activations.mean(axis=1))
                    labels.append(label)
                    groups.append(gid)
                    gid += 1
        X = np.stack(feats)              # [N, n_layers, d]
        return X, np.asarray(labels), np.asarray(groups)

    # -- probe training -----------------------------------------------------

    def _fit_logreg(
        self, x: np.ndarray, y: np.ndarray, l2: float = 1.0, n_iter: int = 300
    ) -> np.ndarray:
        """Minimal L2-regularized logistic regression via gradient descent.

        Kept dependency-light (no sklearn required at runtime) and deterministic.
        Returns the weight vector (bias folded in as the last column removed).
        """

        n, d = x.shape
        w = np.zeros(d)
        b = 0.0
        lr = 0.5
        for _ in range(n_iter):
            z = x @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = x.T @ (p - y) / n + l2 * w / n
            grad_b = float(np.mean(p - y))
            w -= lr * grad_w
            b -= lr * grad_b
        return w, b

    def _cv_layer(
        self, x: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """Grouped k-fold CV accuracy for one layer; also full-data weights."""

        unique_groups = np.unique(groups)
        self.rng.shuffle(unique_groups)
        folds = np.array_split(unique_groups, min(self.n_folds, len(unique_groups)))
        accs = []
        for fold in folds:
            test_mask = np.isin(groups, fold)
            train_mask = ~test_mask
            if train_mask.sum() < 2 or test_mask.sum() < 1:
                continue
            xs, mu, sd = _standardize(x[train_mask])
            w, b = self._fit_logreg(xs, y[train_mask].astype(float))
            xt = (x[test_mask] - mu) / sd
            pred = (xt @ w + b) > 0
            accs.append(float(np.mean(pred == y[test_mask])))
        # Full-data weights for alignment check.
        xs, _, _ = _standardize(x)
        w_full, _ = self._fit_logreg(xs, y.astype(float))
        return (float(np.mean(accs)) if accs else 0.5,
                float(np.std(accs)) if accs else 0.0,
                w_full)

    # -- main entry ---------------------------------------------------------

    def run(
        self, corpus: List[AlignedTuple], horizon: str,
        diff_of_means_direction: Optional[np.ndarray] = None,
    ) -> ProbeResult:
        variant = self.builder.default_variant()
        X, y, groups = self._collect(corpus, horizon, variant)
        n_layers = X.shape[1]
        acc = np.zeros(n_layers)
        acc_std = np.zeros(n_layers)
        weights = np.zeros((n_layers, X.shape[2]))
        for L in range(n_layers):
            acc[L], acc_std[L], weights[L] = self._cv_layer(X[:, L, :], y, groups)

        best_layer = int(np.argmax(acc))
        chance = float(max(np.mean(y), 1.0 - np.mean(y)))

        alignment = 0.0
        if diff_of_means_direction is not None:
            w = weights[best_layer]
            wn = w / (np.linalg.norm(w) + 1e-12)
            dn = diff_of_means_direction / (
                np.linalg.norm(diff_of_means_direction) + 1e-12
            )
            alignment = float(abs(np.dot(wn, dn)))

        return ProbeResult(
            horizon=horizon,
            layer_accuracy=acc,
            layer_accuracy_std=acc_std,
            best_layer=best_layer,
            best_accuracy=float(acc[best_layer]),
            chance=chance,
            weight_alignment=alignment,
            n_samples=int(X.shape[0]),
        )
