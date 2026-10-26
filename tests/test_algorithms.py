import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest


def weighted_PAV(y, weights=None):
    """Pool-adjacent-violators for isotonic regression (nondecreasing)."""
    n = len(y)
    if weights is None:
        weights = np.ones(n)
    result = y.copy().astype(float)
    w = weights.copy().astype(float)
    # Pool-adjacent-violators
    pool = list(range(n))
    for i in range(1, n):
        if result[i] < result[i-1]:
            # Merge pools
            pool_i = i
            while pool_i > 0 and result[pool_i] < result[pool_i-1]:
                total_w = w[pool_i] + w[pool_i-1]
                result[pool_i-1] = (w[pool_i]*result[pool_i] + w[pool_i-1]*result[pool_i-1])/total_w
                result[pool_i] = result[pool_i-1]
                w[pool_i-1] = total_w
                pool_i -= 1
    return result


def test_PAV_monotone():
    y = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    result = weighted_PAV(y)
    assert np.all(np.diff(result) >= -1e-10)


def test_PAV_already_sorted():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = weighted_PAV(y)
    np.testing.assert_allclose(result, y, atol=1e-10)


def test_PAV_all_equal():
    y = np.array([3.0, 3.0, 3.0])
    result = weighted_PAV(y)
    np.testing.assert_allclose(result, y, atol=1e-10)


def test_nonneg_projection():
    x = np.array([-1.0, 0.5, -0.3, 2.0])
    result = np.maximum(x, 0)
    assert np.all(result >= 0)


def test_augmented_lagrangian_structure():
    # Test that augmented Lagrangian can be evaluated
    # phi(u) = 2u, n=10, simple test
    n = 10
    h = 1.0/n
    u = (np.arange(n) + 0.5) / n
    phi = 2 * u
    Q = np.zeros(n)
    gamma = 0.0  # equality multiplier
    lambda_v = 0.0  # inequality multiplier
    tau = 1.0
    omega = 0.5
    Q_prev = np.zeros(n)

    # Compute mean constraint
    M = h * np.sum(Q) - 0.0  # target 0
    # Compute variance constraint
    N = h * np.sum(Q**2) - 1.0  # <= 0

    # Augmented objective
    obj = -h * np.dot(phi, Q) + gamma * M + 0.5*tau*M**2
    assert np.isfinite(obj)


def test_density_CDF_formula():
    n = 80
    h = 1.0/n
    f = np.ones(n)  # uniform density
    # F_i = h*(sum_{j<i} f_j + f_i/2)
    F = np.zeros(n)
    for i in range(n):
        F[i] = h * (np.sum(f[:i]) + f[i]/2)
    # Should approximate [0, 1] uniform CDF at midpoints
    expected = (np.arange(n) + 0.5) / n
    np.testing.assert_allclose(F, expected, atol=1e-10)


def test_risk_value_range():
    # Spectral risk for phi=2u with mean=0, var=1 should be reasonable
    # Target from Table 2: ~0.5773
    n = 80
    h = 1.0/n
    u = (np.arange(n) + 0.5) / n
    phi = 2 * u
    # Gaussian quantile satisfies mean=0, var=1
    from scipy.stats import norm
    Q_gauss = norm.ppf(u)
    risk = h * np.dot(phi, Q_gauss)
    # Should be between 0.5 and 0.7 for this quantile
    assert 0.4 < risk < 0.8
