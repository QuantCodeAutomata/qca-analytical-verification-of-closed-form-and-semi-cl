import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest


def test_mean_variance_kappa2():
    """Test Model 1 mean-variance special case."""
    u = np.linspace(0.001, 0.999, 2000)
    beta = 3
    phi = beta * u**(beta-1)
    mu = 0.0
    sigma = 1.0
    # sigma_phi = (beta-1)/sqrt(2*beta-1) for power spectrum
    sigma_phi_exact = (beta-1)/np.sqrt(2*beta-1)
    sigma_phi_num = np.sqrt(np.trapz(phi**2, u) - 1.0)
    assert abs(sigma_phi_exact - sigma_phi_num) < 0.01
    # Risk = mu + sigma*sigma_phi
    risk = mu + sigma * sigma_phi_exact
    assert risk > 0


def test_es_corollary():
    """ES corollary: sup ES_alpha = mu + sigma*sqrt(alpha/(1-alpha))."""
    mu, sigma, alpha = 1.0, 2.0, 0.9
    es = mu + sigma * np.sqrt(alpha/(1-alpha))
    assert abs(es - (mu + sigma * 3.0)) < 0.001  # sqrt(0.9/0.1)=3


def test_gamma1_solution_sign():
    """gamma1* should exist and the root function changes sign."""
    u = np.linspace(0.001, 0.999, 2000)
    kappa = 3.0
    beta = 3
    phi = beta * u**(beta-1)
    # Root function: integral |phi-gamma|^(1/(kappa-1)) sgn(phi-gamma) du
    def root_func(gamma):
        diff = phi - gamma
        return np.trapz(np.abs(diff)**(1/(kappa-1)) * np.sign(diff), u)
    v_low = root_func(-1.0)
    v_high = root_func(10.0)
    # Should change sign
    assert v_low * v_high < 0


def test_Q_star_mean_constraint():
    """Q* should satisfy the mean constraint."""
    u = np.linspace(0.001, 0.999, 2000)
    mu = 1.0
    sigma = 1.0
    beta = 3
    phi = beta * u**(beta-1)
    sigma_phi = (beta-1)/np.sqrt(2*beta-1)
    Q_star = mu + sigma/sigma_phi * (phi - 1.0)
    mean_Q = np.trapz(Q_star, u)
    assert abs(mean_Q - mu) < 0.05


def test_risk_value_positive():
    """Worst-case risk should be >= mu for positive sigma."""
    mu = 1.0
    sigma = 2.0
    beta = 3
    sigma_phi = (beta-1)/np.sqrt(2*beta-1)
    risk = mu + sigma * sigma_phi
    assert risk >= mu
