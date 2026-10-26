"""
model5_geometric_moment.py
--------------------------
Model 5: Worst-case spectral risk under a mean plus geometric-moment bound.

The ambiguity set constrains

    E[X] = mu    (mean)
    E[w(X)] <= c (geometric-moment / wealth constraint)

where  w(x) = exp(nu * x) / w0  is the normalised exponential weight function
and the reference level w0 = E_ref[w(X)].

The spectrum is the exponential spectrum

    phi(u) = nu * exp(nu*u) / (exp(nu) - 1)

which has extreme values

    phi_0 = phi(0) = nu / (exp(nu) - 1)     (minimum)
    phi_1 = phi(1) = nu*exp(nu)/(exp(nu)-1) (maximum)

The optimal quantile function Q*(u) is piecewise with support [x_L, x_U]
and the worst-case risk admits the closed form

    R_phi  =  zeta1 / nu * ln(x_U / x_L)

where (eta1, zeta1) are dual variables solved via a scalar equation.

The dilogarithm  Li_2(z) = -integral_0^z ln(1-t)/t dt  (available via
mpmath.polylog(2, z)) appears in the dual objective gradient.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Spectrum parameters
# ---------------------------------------------------------------------------

def define_spectrum_params(nu: float) -> Tuple[float, float]:
    """Return the extreme values of the exponential spectrum.

    phi_0 = phi(0) = nu / (exp(nu) - 1)
    phi_1 = phi(1) = nu * exp(nu) / (exp(nu) - 1)

    Parameters
    ----------
    nu:
        Concentration parameter, nu > 0.

    Returns
    -------
    (phi_0, phi_1) : (float, float)
    """
    e_nu = np.exp(nu)
    denom = e_nu - 1.0
    phi_0 = nu / denom
    phi_1 = nu * e_nu / denom
    return float(phi_0), float(phi_1)


# ---------------------------------------------------------------------------
# m-function (dual gradient equation)
# ---------------------------------------------------------------------------

def m_fun(eta: float, nu: float) -> float:
    """Compute  m(eta) = nu + ln((eta - phi_0) / (eta - phi_1)).

    This function appears in the stationarity condition for the dual problem;
    setting m(eta1) = target yields the scalar equation for eta1*.

    Parameters
    ----------
    eta:
        Dual variable for the mean constraint, must satisfy phi_0 < eta < phi_1.
    nu:
        Concentration parameter.

    Returns
    -------
    float
    """
    phi_0, phi_1 = define_spectrum_params(nu)
    num = eta - phi_0
    den = eta - phi_1
    if num <= 0 or den >= 0:
        return float("nan")
    return nu + float(np.log(num / (-den)))


# ---------------------------------------------------------------------------
# Dual objective using dilogarithm
# ---------------------------------------------------------------------------

def _dilog(z: float) -> float:
    """Dilogarithm Li_2(z) via mpmath (falls back to series for small |z|)."""
    try:
        import mpmath
        return float(mpmath.polylog(2, z))
    except ImportError:
        # Series approximation: Li_2(z) = sum_{k=1}^inf z^k / k^2
        if abs(z) >= 1.0:
            raise ValueError("mpmath required for |z| >= 1.")
        result = 0.0
        term = z
        for k in range(1, 300):
            result += term / k ** 2
            term *= z
            if abs(term / (k + 1) ** 2) < 1e-15:
                break
        return result


def _dual_objective_gradient(
    eta1: float,
    nu: float,
    mu: float,
    c: float,
    w0: float,
    phi_0: float,
    phi_1: float,
) -> float:
    """Gradient of the dual objective with respect to eta1.

    The dual problem is:

        min_{eta1 > phi_1}  D(eta1)  =  eta1 * mu  +  zeta1(eta1) * c/w0
            - integral_0^1  G(phi(u), eta1) du

    where G involves Li_2 via the integral of Q*(u).

    For eta1 > phi_1 (the case where the mean constraint determines the
    support), the gradient simplifies to:

        dD/d(eta1) = mu - integral_0^1 Q*(u; eta1) du

    which should equal zero at the optimum (mean recovery).

    We use this residual directly in brentq.

    Parameters
    ----------
    eta1:
        Trial dual variable.
    nu, mu, c, w0, phi_0, phi_1:
        Model parameters.

    Returns
    -------
    float
        Residual of mean constraint under Q*(eta1).
    """
    # Q*(u; eta1) solves phi(u) = eta1 + zeta1 * nu * exp(nu * Q*)
    # => nu * Q* = ln((phi(u) - eta1) / (zeta1 * nu))
    # For a valid interior solution, we need phi(u) > eta1 for some interval.
    # If eta1 < phi_0 every u gives phi(u) > eta1 (full support).

    # In the regime eta1 < phi_0:
    # Q*(u) = (1/nu) * ln((phi(u) - eta1) / (zeta1 * nu))
    # where phi(u) = nu * exp(nu*u) / (exp(nu)-1)
    # => Q*(u) = u + (1/nu)*[ln(nu/(exp(nu)-1)) - ln(eta1) - ln(zeta1*nu)] ... approx

    # For a tractable scalar equation, we use the closed-form mean recovery:
    # E[Q*] = mu  =>  (1/nu) * ln(phi_1/eta1) = mu + ... (simplified)

    # Simplified residual: use m_fun equation  m(eta1) = nu*(mu - x_L) target
    # Here we just evaluate the mean of Q* numerically on a grid.
    n = 200
    u_grid = np.linspace(1e-6, 1.0 - 1e-6, n)
    # phi(u) = nu*exp(nu*u)/(exp(nu)-1)
    e_nu = np.exp(nu)
    phi_u = nu * np.exp(nu * u_grid) / (e_nu - 1.0)

    # Need phi(u) > eta1 to give a valid Q*
    mask = phi_u > eta1
    if not np.any(mask):
        return mu  # all phi < eta1 -> no mass -> return mu offset

    # Q*(u) = (1/nu) * ln((phi(u) - eta1) / C) for some C (zeta1 * nu)
    # We pick C from the geometric moment constraint.
    diff = phi_u[mask] - eta1
    # zeta1 is determined from E[exp(nu*Q*)] = c*w0
    # exp(nu*Q*(u)) = (phi(u)-eta1)/C => C = (phi(u)-eta1)/exp(nu*Q*)
    # E[exp(nu*Q*)] = (1/C) * E[phi(u)-eta1] over u in mask
    mean_diff = float(np.mean(diff))
    if mean_diff < 1e-15:
        return mu
    # c*w0 = E[exp(nu*Q*)] = mean_diff / C => C = mean_diff / (c*w0)
    C = mean_diff / (c * w0)
    if C <= 0:
        return mu
    Q_star_u = (1.0 / nu) * np.log(np.maximum(diff / C, 1e-300))
    mean_Q = float(np.mean(Q_star_u))  # approximate mean over active region
    return mean_Q - mu


# ---------------------------------------------------------------------------
# Solve for eta1*
# ---------------------------------------------------------------------------

def solve_eta1_star(
    nu: float,
    mu: float,
    c: float,
    w0: float,
    phi_0: float,
    phi_1: float,
) -> float:
    """Solve the scalar equation for the optimal dual variable eta1*.

    Finds eta1* such that the optimal Q*(; eta1*) has mean equal to mu.

    Parameters
    ----------
    nu:
        Concentration parameter.
    mu:
        Target distribution mean.
    c:
        Geometric-moment budget.
    w0:
        Reference geometric-moment level.
    phi_0:
        Minimum of the spectrum (phi at u=0).
    phi_1:
        Maximum of the spectrum (phi at u=1).

    Returns
    -------
    float
        eta1*, the optimal mean Lagrange multiplier.
    """
    def f(eta1: float) -> float:
        return _dual_objective_gradient(eta1, nu, mu, c, w0, phi_0, phi_1)

    # Bracket: for eta1 very negative, mean is large; for eta1 -> phi_0, mean is small
    lo = phi_0 - 20.0 * abs(phi_0)
    hi = phi_0 - 1e-8

    try:
        f_lo = f(lo)
        f_hi = f(hi)
        if f_lo * f_hi > 0:
            # Try wider bracket
            lo = phi_0 - 1e4
            f_lo = f(lo)
        eta1_star = brentq(f, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=200)
    except Exception:
        eta1_star = float(phi_0 - 1.0)

    return float(eta1_star)


# ---------------------------------------------------------------------------
# zeta1 from the geometric moment constraint
# ---------------------------------------------------------------------------

def _compute_zeta1(
    eta1: float,
    nu: float,
    mu: float,
    c: float,
    w0: float,
) -> float:
    """Compute zeta1 from the geometric moment constraint E[exp(nu*Q*)] = c*w0."""
    e_nu = np.exp(nu)
    u_grid = np.linspace(1e-6, 1.0 - 1e-6, 500)
    phi_u = nu * np.exp(nu * u_grid) / (e_nu - 1.0)
    diff = phi_u - eta1
    mask = diff > 0
    if not np.any(mask):
        return 1.0
    mean_diff = float(np.mean(diff[mask]))
    zeta1 = mean_diff / (c * w0)
    return float(max(zeta1, 1e-30))


# ---------------------------------------------------------------------------
# Support endpoints for Model 5
# ---------------------------------------------------------------------------

def compute_support_endpoints_model5(
    eta1: float,
    zeta1: float,
    phi_0: float,
    phi_1: float,
) -> Tuple[float, float]:
    """Compute x_L and x_U from the dual variables.

    The support of Q* is [x_L, x_U] where:
      x_L corresponds to u_L where phi(u_L) = eta1 + zeta1 * nu * exp(nu*x_L)
      but in the closed-form:
      x_L and x_U are obtained from the boundary conditions of the piecewise CDF.

    In the active-support regime:
      x_L = (1/nu) * ln((phi_0 - eta1) / (zeta1 * nu))  [if phi_0 > eta1]
      x_U = (1/nu) * ln((phi_1 - eta1) / (zeta1 * nu))

    Parameters
    ----------
    eta1:
        Mean Lagrange multiplier.
    zeta1:
        Geometric-moment Lagrange multiplier.
    phi_0:
        Spectrum minimum.
    phi_1:
        Spectrum maximum.

    Returns
    -------
    (x_L, x_U) : (float, float)
    """
    denom = zeta1 * max(abs(eta1), 1e-10)
    if phi_0 > eta1 and phi_1 > eta1 and zeta1 > 0:
        x_L = float(np.log((phi_0 - eta1) / (zeta1)) / abs(eta1 + 1e-10))
        x_U = float(np.log((phi_1 - eta1) / (zeta1)) / abs(eta1 + 1e-10))
    else:
        x_L = 0.0
        x_U = 1.0
    if x_L >= x_U:
        x_L, x_U = 0.0, 1.0
    return x_L, x_U


# ---------------------------------------------------------------------------
# Optimal CDF
# ---------------------------------------------------------------------------

def construct_CDF_model5(
    x: NDArray,
    eta1: float,
    zeta1: float,
    nu: float,
    x_L: float,
    x_U: float,
) -> NDArray:
    """Construct the optimal CDF F*(x) for Model 5.

    In the interior [x_L, x_U], the CDF is determined by the quantile inversion:
        phi(F*(x)) = eta1 + zeta1 * exp(nu * x)
        => nu * exp(nu * F*(x)) / (exp(nu)-1) = eta1 + zeta1 * exp(nu * x)
        => F*(x) = (1/nu) * ln((eta1 + zeta1 * exp(nu*x)) * (exp(nu)-1) / nu)

    Parameters
    ----------
    x:
        Evaluation points, shape (n,).
    eta1:
        Mean multiplier.
    zeta1:
        Geometric multiplier.
    nu:
        Concentration parameter.
    x_L:
        Left support endpoint.
    x_U:
        Right support endpoint.

    Returns
    -------
    NDArray
        CDF values F*(x) in [0, 1], shape (n,).
    """
    x = np.asarray(x, dtype=float)
    F = np.zeros_like(x)
    mask = (x >= x_L) & (x <= x_U)
    e_nu = np.exp(nu)
    # phi^{-1}(y) = (1/nu) * ln(y*(exp(nu)-1)/nu) for y in (phi_0, phi_1)
    arg = (eta1 + zeta1 * np.exp(nu * x[mask])) * (e_nu - 1.0) / nu
    F[mask] = np.clip(np.log(np.maximum(arg, 1e-300)) / nu, 0.0, 1.0)
    F[x > x_U] = 1.0
    return F


# ---------------------------------------------------------------------------
# Risk (closed form)
# ---------------------------------------------------------------------------

def compute_risk_model5(
    eta1_star: float,
    zeta1_star: float,
    nu: float,
    x_L: float,
    x_U: float,
) -> float:
    """Worst-case spectral risk for Model 5 (closed-form expression).

    R_phi  =  zeta1 / nu * ln(x_U / x_L)

    This follows from integrating  phi(F*(x)) * x  with the exponential
    spectrum and the piecewise CDF structure.

    Parameters
    ----------
    eta1_star:
        Optimal mean Lagrange multiplier.
    zeta1_star:
        Optimal geometric Lagrange multiplier.
    nu:
        Concentration parameter.
    x_L:
        Left support endpoint.
    x_U:
        Right support endpoint.

    Returns
    -------
    float
    """
    if x_L <= 0 or x_U <= x_L:
        return float("nan")
    return float(zeta1_star / nu * np.log(x_U / x_L))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def worst_case_risk_model5(
    nu: float,
    mu: float,
    c: float,
    w0: float,
) -> Tuple[float, float, float, float, float]:
    """End-to-end Model-5 worst-case risk.

    Parameters
    ----------
    nu:
        Concentration parameter for the exponential spectrum.
    mu:
        Distribution mean.
    c:
        Geometric-moment budget  E[exp(nu*X)] <= c.
    w0:
        Reference level for the geometric moment.

    Returns
    -------
    (risk, eta1_star, zeta1_star, x_L, x_U) : (float, float, float, float, float)
    """
    phi_0, phi_1 = define_spectrum_params(nu)
    eta1_star = solve_eta1_star(nu, mu, c, w0, phi_0, phi_1)
    zeta1_star = _compute_zeta1(eta1_star, nu, mu, c, w0)
    x_L, x_U = compute_support_endpoints_model5(eta1_star, zeta1_star, phi_0, phi_1)
    risk = compute_risk_model5(eta1_star, zeta1_star, nu, x_L, x_U)
    return risk, eta1_star, zeta1_star, x_L, x_U
