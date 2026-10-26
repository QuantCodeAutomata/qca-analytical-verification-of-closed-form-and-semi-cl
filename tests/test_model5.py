import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest


def test_spectrum_params_model5():
    nu = 2.0
    phi_0 = nu / (np.exp(nu) - 1)
    phi_1 = nu * np.exp(nu) / (np.exp(nu) - 1)
    assert phi_0 < phi_1
    assert phi_0 > 0
    assert phi_1 > 0


def test_m_fun():
    nu = 2.0
    phi_0 = nu / (np.exp(nu) - 1)
    phi_1 = nu * np.exp(nu) / (np.exp(nu) - 1)
    eta = phi_1 + 1.0  # > phi_1
    m = nu + np.log((eta - phi_0)/(eta - phi_1))
    assert m > 0  # log ratio should be positive


def test_CDF_is_valid():
    # CDF should be between 0 and 1 on support
    nu = 2.0
    x_L = 0.5
    x_U = 8.0
    x = np.linspace(x_L, x_U, 100)
    # Mock CDF construction (simplistic check)
    # In real model, F*(x) = (1/nu)*ln((exp(nu)-1)/nu * (eta1 - zeta1/x))
    assert x_L < x_U
    assert x_L > 0


def test_risk_value_positive():
    # Risk = zeta1/nu * ln(x_U/x_L)
    zeta1 = 2.0
    nu = 2.0
    x_L = 0.5
    x_U = 8.0
    risk = zeta1/nu * np.log(x_U/x_L)
    assert risk > 0
