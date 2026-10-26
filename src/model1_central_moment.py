"""
model1_central_moment.py
------------------------
Model 1: Worst-case spectral risk under a central-moment ambiguity set.

The ambiguity set is

    M1(mu, kappa, m_kappa) = { Q non-decreasing on [0,1] :
        integral_0^1 Q(u) du = mu,
        integral_0^1 |Q(u) - mu|^kappa du <= m_kappa }

and the programme solved is

    sup_{Q in M1}  R_phi(Q)  =  integral_0^1 phi(u) Q(u) du.

By first-order optimality, the optimiser satisfies

    Q*(u) = mu + lambda1 * sgn(phi(u) - gamma1) * |phi(u) - gamma1|^{1/(kappa-1)}

where gamma1 is the unique root of

    T(gamma) := integral_0^1 |phi(u)-gamma|^{1/(kappa-1)} sgn(phi(u)-gamma) du = 0

and lambda1 is chosen so that the moment constraint is tight.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq

from src.spectral_risk import risk_quantile_space


# ---------------------------------------------------------------------------
# Step 1 – solve for gamma1
# ---------------------------------------------------------------------------

def _T_gamma(gamma: float, phi_vals: NDArray, u_grid: NDArray, kappa: float) -> float:
    """Integrand-sum for the stationarity equation T(gamma) = 0."""
    diff = phi_vals - gamma
    exp = 1.0 / (kappa - 1.0)
    integrand = np.abs(diff) ** exp * np.sign(diff)
    return float(np.trapz(integrand, u_grid))


def solve_gamma1(
    phi_vals: NDArray,
    u_grid: NDArray,
    kappa: float,
) -> float:
    """Find gamma1 satisfying  integral_0^1 |phi(u)-gamma|^{1/(kappa-1)} sgn(phi(u)-gamma) du = 0.

    Parameters
    ----------
    phi_vals:
        Spectrum phi evaluated on *u_grid*, shape (n,).
    u_grid:
        Probability grid in [0, 1], shape (n,).
    kappa:
        Moment order, kappa > 1.

    Returns
    -------
    float
        gamma1, the unique root of the stationarity equation.
    """
    phi_min = float(phi_vals.min())
    phi_max = float(phi_vals.max())
    # T is strictly decreasing in gamma; bracket between extremes of phi
    lo = phi_min - 1e-6
    hi = phi_max + 1e-6

    T_lo = _T_gamma(lo, phi_vals, u_grid, kappa)
    T_hi = _T_gamma(hi, phi_vals, u_grid, kappa)
    if T_lo * T_hi > 0:
        # Fall back: bisect a wider range
        lo, hi = phi_min - 10.0, phi_max + 10.0

    gamma1 = brentq(_T_gamma, lo, hi, args=(phi_vals, u_grid, kappa), xtol=1e-12, rtol=1e-12)
    return float(gamma1)


# ---------------------------------------------------------------------------
# Step 2 – compute lambda1
# ---------------------------------------------------------------------------

def compute_lambda1(
    phi_vals: NDArray,
    u_grid: NDArray,
    gamma1: float,
    kappa: float,
    m_kappa: float,
) -> float:
    """Compute the Lagrange multiplier lambda1 from the binding moment constraint.

    The constraint  integral_0^1 |Q*(u) - mu|^kappa du = m_kappa  together with

        Q*(u) - mu  =  lambda1 * |phi(u) - gamma1|^{1/(kappa-1)} * sgn(phi(u) - gamma1)

    gives

        lambda1^kappa * C  =  m_kappa,   C = integral_0^1 |phi-gamma1|^{kappa/(kappa-1)} du.

    Parameters
    ----------
    phi_vals:
        Spectrum values on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    gamma1:
        Stationarity root from :func:`solve_gamma1`.
    kappa:
        Moment order.
    m_kappa:
        Moment budget (right-hand side of the constraint).

    Returns
    -------
    float
        lambda1 >= 0.
    """
    exp = kappa / (kappa - 1.0)
    C = float(np.trapz(np.abs(phi_vals - gamma1) ** exp, u_grid))
    if C < 1e-15:
        return 0.0
    lambda1 = (m_kappa / C) ** (1.0 / kappa)
    return float(lambda1)


# ---------------------------------------------------------------------------
# Step 3 – construct Q*
# ---------------------------------------------------------------------------

def construct_Q_star(
    phi_vals: NDArray,
    u_grid: NDArray,
    mu: float,
    kappa: float,
    lambda1: float,
    gamma1: float,
) -> NDArray:
    """Build the worst-case quantile function Q*(u) on the given grid.

    Q*(u) = mu + lambda1 * sgn(phi(u) - gamma1) * |phi(u) - gamma1|^{1/(kappa-1)}

    Parameters
    ----------
    phi_vals:
        Spectrum on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    mu:
        Mean of the distribution.
    kappa:
        Moment order.
    lambda1:
        Moment multiplier from :func:`compute_lambda1`.
    gamma1:
        Stationarity root from :func:`solve_gamma1`.

    Returns
    -------
    NDArray
        Q*(u) on *u_grid*, shape (n,).  Non-decreasing by construction when
        phi is non-decreasing (standard spectral-measure condition).
    """
    diff = phi_vals - gamma1
    exp = 1.0 / (kappa - 1.0)
    return mu + lambda1 * np.sign(diff) * np.abs(diff) ** exp


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def worst_case_risk_model1(
    phi_func: Callable[..., NDArray],
    u_grid: NDArray,
    mu: float,
    kappa: float,
    m_kappa: float,
    **phi_kwargs,
) -> float:
    """Compute the worst-case spectral risk over the Model-1 ambiguity set.

    Parameters
    ----------
    phi_func:
        Spectrum function with signature phi(u, **phi_kwargs).
    u_grid:
        Probability grid in [0, 1], shape (n,).
    mu:
        Mean constraint.
    kappa:
        Moment order, kappa > 1.
    m_kappa:
        Moment budget.
    **phi_kwargs:
        Additional keyword arguments forwarded to *phi_func*.

    Returns
    -------
    float
        Worst-case risk value R_phi(Q*).
    """
    phi_vals = np.asarray(phi_func(u_grid, **phi_kwargs), dtype=float)
    gamma1 = solve_gamma1(phi_vals, u_grid, kappa)
    lambda1 = compute_lambda1(phi_vals, u_grid, gamma1, kappa, m_kappa)
    Q_star = construct_Q_star(phi_vals, u_grid, mu, kappa, lambda1, gamma1)
    return risk_quantile_space(phi_vals, Q_star, u_grid)


# ---------------------------------------------------------------------------
# Special cases
# ---------------------------------------------------------------------------

def mean_variance_special_case(
    phi_func: Callable[..., NDArray],
    u_grid: NDArray,
    mu: float,
    sigma: float,
    **phi_kwargs,
) -> Tuple[NDArray, float]:
    """Kappa=2 closed-form: Q*(u) = mu + (sigma/sigma_phi)*(phi(u) - 1).

    The moment constraint  integral_0^1 (Q(u)-mu)^2 du <= sigma^2  is used.
    Since integral_0^1 phi(u) du = 1, we have gamma1 = 1 and

        Q*(u)  =  mu + (sigma / sigma_phi) * (phi(u) - 1)

    where  sigma_phi = sqrt(integral_0^1 (phi(u)-1)^2 du).

    Parameters
    ----------
    phi_func:
        Spectrum function.
    u_grid:
        Probability grid in [0, 1].
    mu:
        Distribution mean.
    sigma:
        Standard deviation budget (sqrt of moment budget m_2).
    **phi_kwargs:
        Forwarded to *phi_func*.

    Returns
    -------
    (Q_star, risk) : (NDArray, float)
        Worst-case quantile function and risk value.
    """
    phi_vals = np.asarray(phi_func(u_grid, **phi_kwargs), dtype=float)
    sigma_phi = float(np.sqrt(np.trapz((phi_vals - 1.0) ** 2, u_grid)))
    Q_star = mu + (sigma / sigma_phi) * (phi_vals - 1.0)
    risk = float(np.trapz(phi_vals * Q_star, u_grid))
    return Q_star, risk


def es_corollary(mu: float, sigma: float, alpha: float) -> float:
    """Closed-form worst-case ES under mean-variance ambiguity (kappa=2).

    For the ES_alpha spectrum  phi(u) = (1-alpha)^{-1} * 1_{u >= alpha},
    the worst-case risk over  { Q : mean = mu, Var <= sigma^2 }  equals

        sup ES_alpha  =  mu  +  sigma * sqrt(alpha / (1 - alpha)).

    Parameters
    ----------
    mu:
        Distribution mean.
    sigma:
        Standard deviation bound.
    alpha:
        ES confidence level in (0, 1).

    Returns
    -------
    float
        Worst-case ES value.
    """
    return mu + sigma * np.sqrt(alpha / (1.0 - alpha))


# ---------------------------------------------------------------------------
# Numerical cross-check via cvxpy
# ---------------------------------------------------------------------------

def cvxpy_crosscheck_model1(
    phi_func: Callable[..., NDArray],
    mu: float,
    kappa: float,
    m_kappa: float,
    n: int = 200,
    **phi_kwargs,
) -> dict:
    """Validate :func:`worst_case_risk_model1` against a cvxpy optimisation.

    Formulates the discrete LP/CP

        max  h * phi^T Q
        s.t. h * 1^T Q = mu
             h * sum_i |Q_i - mu|^kappa <= m_kappa  (power-cone)
             Q_1 <= Q_2 <= ... <= Q_n

    using cvxpy with a power-cone solver.

    Parameters
    ----------
    phi_func:
        Spectrum function.
    mu:
        Mean constraint.
    kappa:
        Moment order.
    m_kappa:
        Moment budget.
    n:
        Number of grid points.
    **phi_kwargs:
        Forwarded to *phi_func*.

    Returns
    -------
    dict with keys 'analytic', 'cvxpy', 'abs_diff'.
    """
    import cvxpy as cp

    h = 1.0 / n
    u_grid = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    phi_vals = np.asarray(phi_func(u_grid, **phi_kwargs), dtype=float)

    analytic = worst_case_risk_model1(phi_func, u_grid, mu, kappa, m_kappa, **phi_kwargs)

    Q = cp.Variable(n)
    objective = cp.Maximize(h * (phi_vals @ Q))
    constraints = [
        h * cp.sum(Q) == mu,
        Q[:-1] <= Q[1:],
    ]

    if abs(kappa - 2.0) < 1e-10:
        # Variance constraint via SOC
        constraints.append(cp.sum_squares(Q - mu) * h <= m_kappa)
    else:
        # General power-cone: sum_i t_i <= m_kappa / h, |Q_i - mu| <= t_i^{1/kappa}
        # Use cp.power for elementwise power cone
        t = cp.Variable(n, nonneg=True)
        s = cp.Variable(n)
        for i in range(n):
            constraints.append(s[i] == Q[i] - mu)
            constraints.append(cp.norm(cp.vstack([s[i]]), 2) <= t[i])
        constraints.append(h * cp.sum(t) <= m_kappa)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    cvxpy_val = float(prob.value) if prob.value is not None else float("nan")
    return {
        "analytic": analytic,
        "cvxpy": cvxpy_val,
        "abs_diff": abs(analytic - cvxpy_val),
    }
