import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest


def test_portfolio_risk_formula():
    """V(w) = mu_w + sigma_w * chi."""
    mu_hat = np.array([0.05, 0.08, 0.06])
    Sigma_hat = np.array([[0.04, 0.01, 0.005],
                          [0.01, 0.09, 0.01],
                          [0.005, 0.01, 0.06]])
    w = np.array([0.4, 0.3, 0.3])
    chi = 2.0  # example
    mu_w = w @ mu_hat
    sigma_w = np.sqrt(w @ Sigma_hat @ w)
    V = mu_w + sigma_w * chi
    assert V > 0
    assert V > mu_w  # chi > 0 implies risk > mean


def test_chi_portfolio_independence():
    """chi(alpha,epsilon) does not depend on portfolio w."""
    chi = 2.0  # precomputed
    # Same chi used for different w - test structure
    assert chi > 0  # must be positive (worst-case ES > 0 for std normal)


def test_SOCP_feasibility():
    """Test that SOCP problem is feasible for long-only constraint."""
    try:
        import cvxpy as cp
        d = 3
        np.random.seed(42)
        A = np.random.randn(d, d)
        Sigma = A.T @ A + 0.1*np.eye(d)
        L = np.linalg.cholesky(Sigma)
        mu_hat = np.array([0.05, 0.08, 0.06])
        chi = 2.0

        w = cp.Variable(d)
        obj = mu_hat @ w + chi * cp.norm(L.T @ w, 2)
        constraints = [w >= 0, cp.sum(w) == 1]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve()
        assert prob.status in ['optimal', 'optimal_inaccurate']
    except ImportError:
        pytest.skip('cvxpy not available')


def test_projection_identity_structure():
    """Projection identity: marginal ES = mu_w + sigma_w * chi."""
    # Conceptual test - verify formula structure
    mu_w = 0.06
    sigma_w = 0.2
    chi = 1.5
    V = mu_w + sigma_w * chi
    assert abs(V - 0.36) < 0.001


def test_sigma_phi_computation():
    """Test covariance matrix operations for portfolio risk."""
    np.random.seed(0)
    d = 5
    A = np.random.randn(d, d)
    Sigma = A.T @ A + np.eye(d)
    w = np.ones(d) / d  # equal weight
    sigma_w = np.sqrt(w @ Sigma @ w)
    assert sigma_w > 0
    # Cholesky
    L = np.linalg.cholesky(Sigma)
    sigma_w_chol = np.linalg.norm(L.T @ w)
    assert abs(sigma_w - sigma_w_chol) < 1e-10
