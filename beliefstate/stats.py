"""Statistical inference utilities.

The paper's claims are comparative ("belief contrasts separate more than null
contrasts", "steering along b shifts behaviour more than random r"). To report
these at ICLR standard we attach:

    * permutation tests  -- a distribution-free p-value for whether an observed
      separation could arise under the null of exchangeable labels;
    * bootstrap CIs      -- nonparametric confidence intervals for effect sizes
      (e.g. the belief-vs-random intervention gap);
    * Cohen's d          -- a standardized effect size for the same comparison.

All routines take an explicit ``numpy`` Generator so results are reproducible
under the global seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np


@dataclass
class TestResult:
    statistic: float
    p_value: float
    n_permutations: int

    def as_dict(self) -> dict:
        return {
            "statistic": round(float(self.statistic), 5),
            "p_value": round(float(self.p_value), 5),
            "n_permutations": int(self.n_permutations),
        }


@dataclass
class CIResult:
    estimate: float
    low: float
    high: float
    level: float

    def as_dict(self) -> dict:
        return {
            "estimate": round(float(self.estimate), 5),
            "ci_low": round(float(self.low), 5),
            "ci_high": round(float(self.high), 5),
            "level": self.level,
        }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference between two samples (pooled SD).

    Returns NaN when the pooled SD collapses relative to the mean difference.
    That regime is *not* a huge effect size -- it means the comparison is
    degenerate (e.g. the readout axis coincides with the intervention axis, so
    within-group variance is ~0 by construction), and a standardized effect size
    is undefined. We surface NaN rather than a capped sentinel so the degeneracy
    is visible instead of being dressed up as "d = 100".
    """

    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    mean_diff = a.mean() - b.mean()
    if pooled < 1e-8 * (abs(mean_diff) + 1e-8):
        return float("nan")
    return float(mean_diff / pooled)


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    rng: np.random.Generator,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_permutations: int = 5000,
    alternative: str = "greater",
) -> TestResult:
    """Two-sample permutation test.

    Default statistic is the difference in means; ``alternative`` is one of
    ``greater`` / ``less`` / ``two-sided`` referring to group_a vs group_b.
    """

    a = np.asarray(group_a, float)
    b = np.asarray(group_b, float)
    if statistic is None:
        statistic = lambda x, y: float(x.mean() - y.mean())  # noqa: E731

    observed = statistic(a, b)
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_stat = statistic(pooled[:na], pooled[na:])
        if alternative == "greater":
            count += perm_stat >= observed
        elif alternative == "less":
            count += perm_stat <= observed
        else:  # two-sided
            count += abs(perm_stat) >= abs(observed)
    # +1 smoothing so the p-value is never exactly zero.
    p = (count + 1) / (n_permutations + 1)
    return TestResult(statistic=observed, p_value=p, n_permutations=n_permutations)


def bootstrap_ci(
    sample: np.ndarray,
    rng: np.random.Generator,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_boot: int = 5000,
    level: float = 0.95,
) -> CIResult:
    """Percentile bootstrap confidence interval for a scalar statistic."""

    x = np.asarray(sample, float)
    if statistic is None:
        statistic = lambda v: float(v.mean())  # noqa: E731
    n = len(x)
    estimate = statistic(x)
    if n < 2:
        return CIResult(estimate, estimate, estimate, level)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = statistic(x[idx])
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(boots, [alpha, 1.0 - alpha])
    return CIResult(estimate=estimate, low=float(low), high=float(high), level=level)


def paired_bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 5000,
    level: float = 0.95,
) -> CIResult:
    """Bootstrap CI for the mean *paired* difference a - b."""

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff = a - b
    return bootstrap_ci(diff, rng, n_boot=n_boot, level=level)


# ---------------------------------------------------------------------------
# Cluster-aware inference
# ---------------------------------------------------------------------------
#
# The experimental units are *clustered*: contrast pairs and intervention
# instances drawn from the same firm/period are correlated, so treating the
# individual projections as i.i.d. under-counts the real uncertainty and yields
# anti-conservative p-values and too-narrow CIs. The routines below resample at
# the *cluster* level, which is the honest unit of replication here.


def effective_sample_size(values: np.ndarray, clusters: np.ndarray) -> float:
    """Design-effect-corrected effective N given a one-way clustering.

    n_eff = n / (1 + (m_bar - 1) * ICC), where ICC is the intra-class
    correlation from a one-way ANOVA decomposition and m_bar is the average
    cluster size. Returns n when there is effectively no clustering.
    """

    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    n = len(values)
    uniq = np.unique(clusters)
    k = len(uniq)
    if k < 2 or n <= k:
        return float(n)
    grand = values.mean()
    sizes = np.array([np.sum(clusters == c) for c in uniq], float)
    means = np.array([values[clusters == c].mean() for c in uniq])
    ss_between = float(np.sum(sizes * (means - grand) ** 2))
    ss_within = float(np.sum([
        np.sum((values[clusters == c] - values[clusters == c].mean()) ** 2)
        for c in uniq
    ]))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    m0 = (n - np.sum(sizes ** 2) / n) / (k - 1)  # ANOVA cluster-size constant
    var_between = max(0.0, (ms_between - ms_within) / max(m0, 1e-12))
    icc = var_between / (var_between + ms_within + 1e-12)
    m_bar = n / k
    deff = 1.0 + (m_bar - 1.0) * icc
    return float(n / max(deff, 1e-12))


def cluster_permutation_test(
    values_a: np.ndarray,
    clusters_a: np.ndarray,
    values_b: np.ndarray,
    clusters_b: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int = 5000,
    alternative: str = "greater",
) -> TestResult:
    """Permutation test that permutes whole *clusters* between groups.

    Each cluster is exchangeable as a unit; we shuffle cluster-to-group
    assignments (stratified so group A keeps its original number of clusters)
    and recompute the difference of the pooled per-observation means. This keeps
    the test valid under within-cluster correlation, unlike shuffling
    individual observations.
    """

    va = np.asarray(values_a, float)
    vb = np.asarray(values_b, float)
    ca = np.asarray(clusters_a)
    cb = np.asarray(clusters_b)

    # Build a per-cluster table. Cluster ids in A and B are namespaced so an id
    # colliding across groups is not accidentally merged.
    clusters, obs = [], []
    for cid in np.unique(ca):
        clusters.append(("A", cid)); obs.append(va[ca == cid])
    for cid in np.unique(cb):
        clusters.append(("B", cid)); obs.append(vb[cb == cid])
    n_a = len(np.unique(ca))
    n_clusters = len(clusters)

    def pooled_mean_diff(assign_a):
        a_obs = np.concatenate([obs[i] for i in assign_a]) if assign_a else np.array([])
        b_idx = [i for i in range(n_clusters) if i not in assign_a]
        b_obs = np.concatenate([obs[i] for i in b_idx]) if b_idx else np.array([])
        if a_obs.size == 0 or b_obs.size == 0:
            return 0.0
        return float(a_obs.mean() - b_obs.mean())

    observed = float(va.mean() - vb.mean())
    idx_all = np.arange(n_clusters)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(idx_all)
        assign_a = list(perm[:n_a])
        stat = pooled_mean_diff(assign_a)
        if alternative == "greater":
            count += stat >= observed
        elif alternative == "less":
            count += stat <= observed
        else:
            count += abs(stat) >= abs(observed)
    p = (count + 1) / (n_permutations + 1)
    return TestResult(statistic=observed, p_value=p, n_permutations=n_permutations)


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    rng: np.random.Generator,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_boot: int = 5000,
    level: float = 0.95,
) -> CIResult:
    """Percentile bootstrap that resamples whole clusters with replacement.

    The cluster is the resampling unit, so the CI reflects between-cluster
    variability rather than the (inflated) per-observation count.
    """

    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    if statistic is None:
        statistic = lambda v: float(v.mean())  # noqa: E731
    uniq = np.unique(clusters)
    groups = [values[clusters == c] for c in uniq]
    k = len(uniq)
    estimate = statistic(values)
    if k < 2:
        return CIResult(estimate, estimate, estimate, level)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, size=k)
        sample = np.concatenate([groups[j] for j in pick])
        boots[i] = statistic(sample)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(boots, [alpha, 1.0 - alpha])
    return CIResult(estimate=estimate, low=float(low), high=float(high), level=level)


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down correction for a family of p-values.

    Returns the family-adjusted p-values (in the original order) and the
    reject/keep decisions at level ``alpha``. Controls the family-wise error
    rate across the layer x horizon search without assuming independence.
    """

    p = np.asarray(list(p_values), float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)  # enforce monotonicity
        adjusted[idx] = min(running, 1.0)
    reject = adjusted <= alpha
    return {
        "p_adjusted": adjusted.tolist(),
        "reject": reject.tolist(),
        "n_tests": int(m),
        "alpha": alpha,
    }
