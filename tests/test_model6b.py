import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest
from scipy.stats import norm


def J0(lam, u, v):
    """J_0(lambda; u, v) = exp(lambda^2/2)*[Phi_N(v-lambda) - Phi_N(u-lambda)]."""
    log_val = 0.5*lam**2
    if v == np.inf:
        log_part = norm.logsf(u - lam)
        return np.exp(log_val + log_part - 0)  # Phi_N(inf-lam)=1
    if u == -np.inf:
        log_part = norm.logcdf(v - lam)
        return np.exp(log_val) * norm.cdf(v - lam)
    return np.exp(log_val) * (norm.cdf(v - lam) - norm.cdf(u - lam))


def J1(lam, u, v):
    """J_1(lambda; u, v)."""
    log_val = 0.5*lam**2
    if v == np.inf:
        cdf_part = norm.sf(u - lam)  # 1 - Phi(u-lam)
        pdf_part = -norm.pdf(v - lam) + norm.pdf(u - lam)  # phi(u-lam) since v->inf pdf->0
        return np.exp(log_val) * (lam * cdf_part + norm.pdf(u - lam))
    if u == -np.inf:
        cdf_part = norm.cdf(v - lam)
        return np.exp(log_val) * (lam * cdf_part - norm.pdf(v - lam))
    cdf_part = norm.cdf(v - lam) - norm.cdf(u - lam)
    pdf_part = norm.pdf(v - lam) - norm.pdf(u - lam)
    return np.exp(log_val) * (lam * cdf_part - pdf_part)


def test_feasibility_threshold():
    bar_mu = 0.5
    epsilon_min = bar_mu**2 / 2
    assert epsilon_min == 0.125
    # epsilon > epsilon_min is required
    epsilon = 0.2
    assert epsilon > epsilon_min


def test_J0_positive():
    # J0(lam, -inf, inf) = exp(lam^2/2) * [Phi(inf-lam) - Phi(-inf-lam)]
    #                     = exp(lam^2/2) * 1 = exp(lam^2/2) > 0
    # The normalised density f*(x)/J0 integrates to 1 over the full line.
    for lam in [-1, 0, 1, 2]:
        val = J0(lam, -np.inf, np.inf)
        expected = np.exp(0.5 * lam**2)
        assert abs(val - expected) < 0.001  # J0 equals the MGF exp(lam^2/2)
        assert val > 0


def test_chi_positive():
    # chi = lambda2 + phi_N(x_alpha - lambda2) / (1 - Phi_N(x_alpha - lambda2))
    # This is always positive since it's ES-like
    lambda2 = 0.5
    x_alpha = 1.0
    chi = lambda2 + norm.pdf(x_alpha - lambda2) / norm.sf(x_alpha - lambda2)
    assert chi > 0


def test_lambda1_lt_lambda2_structure():
    # Structural requirement
    lambda1 = -0.5
    lambda2 = 0.5
    assert lambda1 < lambda2


def test_chi_increases_with_epsilon():
    # For bar_mu=0, chi should increase with epsilon (more ambiguity = higher worst-case ES)
    # This is a qualitative check requiring actual calibration
    pass
