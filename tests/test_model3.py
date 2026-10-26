import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest


def test_sigma_phi_power_spectrum():
    """sigma_phi=(beta-1)/sqrt(2*beta-1) for power spectrum."""
    for beta in [2, 3, 5]:
        sigma_phi_exact = (beta-1)/np.sqrt(2*beta-1)
        u = np.linspace(0.001, 0.999, 5000)
        phi = beta * u**(beta-1)
        sigma_phi_num = np.sqrt(np.trapz(phi**2, u) - 1.0)
        assert abs(sigma_phi_exact - sigma_phi_num) < 0.01


def test_M_alpha():
    mu, sigma, alpha = 3.0, 10.0, 0.9
    M_alpha = mu + sigma * np.sqrt(alpha/(1-alpha))
    assert abs(M_alpha - (3 + 10*3)) < 0.001


def test_benchmark_ES():
    u = np.linspace(0.001, 0.999, 5000)
    alpha = 0.9
    beta = 3
    mu, sigma = 3.0, 10.0
    phi = beta * u**(beta-1)
    sigma_phi = (beta-1)/np.sqrt(2*beta-1)
    A0 = np.trapz(phi[u>=alpha] - 1, u[u>=alpha]) / (1-alpha)
    bar_c = mu + sigma * A0 / sigma_phi
    # Paper reports bar_c ~ 22.117
    assert abs(bar_c - 22.117) < 0.5


def test_Q_diamond_mean():
    u = np.linspace(0.001, 0.999, 5000)
    beta = 3
    mu, sigma = 3.0, 10.0
    phi = beta * u**(beta-1)
    sigma_phi = (beta-1)/np.sqrt(2*beta-1)
    Q_diamond = mu + sigma/sigma_phi * (phi - 1)
    mean_Q = np.trapz(Q_diamond, u)
    assert abs(mean_Q - mu) < 0.1


def test_variance_Q_diamond():
    u = np.linspace(0.001, 0.999, 5000)
    beta = 3
    mu, sigma = 3.0, 10.0
    phi = beta * u**(beta-1)
    sigma_phi = (beta-1)/np.sqrt(2*beta-1)
    Q_diamond = mu + sigma/sigma_phi * (phi - 1)
    var_Q = np.trapz((Q_diamond - mu)**2, u)
    assert abs(var_Q - sigma**2) < 1.0
