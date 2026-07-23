"""Tests for the statistical-inference utilities (stats.py).

These check the estimators behave correctly on constructed inputs with known
answers, that the degenerate "perfect separation" case (pooled SD -> 0) returns
NaN rather than a capped sentinel, and that the cluster-aware routines account
for within-cluster correlation.
"""

from __future__ import annotations

import numpy as np

from beliefstate.stats import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    cluster_permutation_test,
    cohens_d,
    effective_sample_size,
    holm_bonferroni,
    paired_bootstrap_diff_ci,
    permutation_test,
)


def test_cohens_d_known_value():
    # Two samples one pooled-SD apart should give |d| ~ 1.
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 2000)
    b = rng.normal(0.0, 1.0, 2000)
    d = cohens_d(a, b)
    assert 0.85 < d < 1.15


def test_cohens_d_nan_on_perfect_separation():
    # Constant samples -> pooled SD 0 relative to the mean gap. The effect size
    # is undefined; we surface NaN so the degeneracy is visible, not a "d=100".
    a = np.ones(50)
    b = np.zeros(50)
    assert np.isnan(cohens_d(a, b))


def test_cohens_d_nan_when_identical():
    # Zero mean difference with zero variance is also degenerate -> NaN.
    a = np.ones(50)
    b = np.ones(50)
    assert np.isnan(cohens_d(a, b))


def test_cohens_d_finite_with_small_but_real_variance():
    rng = np.random.default_rng(11)
    a = rng.normal(1.0, 0.1, 200)
    b = rng.normal(0.0, 0.1, 200)
    d = cohens_d(a, b)
    assert np.isfinite(d) and d > 5.0  # large but well-defined


def test_cohens_d_too_few_samples():
    assert np.isnan(cohens_d(np.array([1.0]), np.array([0.0])))


def test_permutation_test_detects_separation():
    rng = np.random.default_rng(1)
    a = rng.normal(3.0, 1.0, 100)
    b = rng.normal(0.0, 1.0, 100)
    res = permutation_test(a, b, rng, n_permutations=2000, alternative="greater")
    assert res.statistic > 0
    assert res.p_value < 0.01
    assert res.n_permutations == 2000


def test_permutation_test_null_is_nonsignificant():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(0.0, 1.0, 200)
    res = permutation_test(a, b, rng, n_permutations=2000, alternative="two-sided")
    assert res.p_value > 0.05


def test_permutation_pvalue_never_zero():
    rng = np.random.default_rng(3)
    a = np.full(30, 10.0)
    b = np.full(30, -10.0)
    res = permutation_test(a, b, rng, n_permutations=500)
    assert res.p_value > 0.0  # +1 smoothing


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(5.0, 2.0, 500)
    ci = bootstrap_ci(x, rng, n_boot=2000, level=0.95)
    assert ci.low < ci.estimate < ci.high
    assert ci.low < 5.0 < ci.high


def test_bootstrap_ci_degenerate_single_sample():
    ci = bootstrap_ci(np.array([7.0]), np.random.default_rng(5))
    assert ci.low == ci.high == ci.estimate == 7.0


def test_paired_bootstrap_diff_ci():
    rng = np.random.default_rng(6)
    a = rng.normal(2.0, 0.5, 400)
    b = rng.normal(0.0, 0.5, 400)
    ci = paired_bootstrap_diff_ci(a, b, rng, n_boot=2000)
    # Difference is centered near 2 and CI should exclude 0.
    assert ci.low > 0.0
    assert 1.7 < ci.estimate < 2.3


def test_result_as_dict_roundtrips():
    rng = np.random.default_rng(7)
    res = permutation_test(np.ones(10), np.zeros(10), rng, n_permutations=100)
    d = res.as_dict()
    assert set(d) == {"statistic", "p_value", "n_permutations"}
    ci = bootstrap_ci(np.arange(10.0), rng).as_dict()
    assert set(ci) == {"estimate", "ci_low", "ci_high", "level"}


# ---------------------------------------------------------------------------
# Cluster-aware inference
# ---------------------------------------------------------------------------


def test_effective_n_shrinks_under_strong_clustering():
    # Perfectly correlated within clusters: every observation in a cluster is
    # identical, so the effective N should collapse toward the cluster count.
    clusters = np.repeat(np.arange(10), 20)          # 10 clusters x 20 obs
    values = np.repeat(np.arange(10, dtype=float), 20)  # constant within cluster
    n_eff = effective_sample_size(values, clusters)
    assert n_eff < 20.0            # far below the nominal 200
    assert n_eff <= len(values)


def test_effective_n_near_nominal_without_clustering():
    rng = np.random.default_rng(20)
    clusters = np.repeat(np.arange(10), 20)
    values = rng.normal(0.0, 1.0, 200)  # cluster label carries no signal
    n_eff = effective_sample_size(values, clusters)
    assert n_eff > 100.0  # little design-effect inflation


def test_cluster_permutation_less_significant_than_iid_when_clustered():
    # Signal lives entirely between clusters: an i.i.d. test sees 2*n_obs
    # "independent" points and is over-confident; the cluster test, with only a
    # handful of exchangeable clusters, should yield a substantially larger p.
    rng = np.random.default_rng(21)
    ca = np.repeat(np.arange(4), 25)
    cb = np.repeat(np.arange(4, 8), 25)
    a = np.repeat(rng.normal(1.0, 0.05, 4), 25)
    b = np.repeat(rng.normal(0.0, 0.05, 4), 25)
    iid = permutation_test(a, b, np.random.default_rng(1), n_permutations=2000)
    clustered = cluster_permutation_test(
        a, ca, b, cb, np.random.default_rng(1), n_permutations=2000
    )
    assert clustered.p_value > iid.p_value


def test_cluster_bootstrap_ci_wider_than_iid_when_clustered():
    rng = np.random.default_rng(22)
    clusters = np.repeat(np.arange(6), 30)
    values = np.repeat(rng.normal(0.0, 1.0, 6), 30)  # constant within cluster
    iid = bootstrap_ci(values, np.random.default_rng(3))
    clustered = cluster_bootstrap_ci(values, clusters, np.random.default_rng(3))
    assert (clustered.high - clustered.low) > (iid.high - iid.low)


def test_holm_bonferroni_monotone_and_controls_family():
    p = [0.001, 0.02, 0.03, 0.5]
    out = holm_bonferroni(p, alpha=0.05)
    adj = out["p_adjusted"]
    # Adjusted p-values are >= raw and never exceed 1.
    assert all(a >= r - 1e-12 for a, r in zip(adj, p))
    assert all(a <= 1.0 for a in adj)
    assert out["n_tests"] == 4
    # Smallest raw p (0.001 * 4 = 0.004) stays significant; largest does not.
    assert out["reject"][0] is True
    assert out["reject"][3] is False


def test_holm_bonferroni_all_null():
    out = holm_bonferroni([0.9, 0.8, 0.95], alpha=0.05)
    assert not any(out["reject"])
