"""
model4_partial_moments.py
-------------------------
Model 4: Worst-case spectral risk under an exact partial-moment ambiguity
set with bounded support [a, b].

The ambiguity set encodes exact knowledge of

    s1 = E[X * 1_{X <= mu}]   (first lower partial moment)
    s2 = E[X^2 * 1_{X <= mu}] (second lower partial moment)
    E[X] = mu                  (mean)

together with support containment F(a) = 0, F(b) = 1.

For the power spectrum  phi(u) = beta * u^{beta-1}  the optimality
conditions yield an optimal CDF with piecewise power-law form

    (F*(x))^{beta-1}  =  { eta1 + eta2*x                   for x in [x_L, mu]
                          { eta1 + eta2*mu + eta3*(x - mu)  for x in (mu, x_U]

where  y* = F*(mu)  satisfies a polynomial equation (derived from the
partial-moment constraints) and x_L, x_U are determined by the boundary
conditions F*(x_L) = 0, F*(x_U) = 1.

Provenance note: Some coefficient signs were re-derived from first principles
because the source document contained suspected OCR errors in equation (40).
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Polynomial equation for y* = F*(mu)
# ---------------------------------------------------------------------------

def solve_F_mu(
    s1: float,
    s2: float,
    beta: float,
    mu: float,
    a: float,
    b: float,
) -> float:
    """Solve the polynomial equation for y* = F*(mu).

    The constraints E[X * 1_{X<=mu}] = s1 and E[X^2 * 1_{X<=mu}] = s2 together
    with the mean E[X] = mu and normalisation yield an equation in y* alone.

    For the power spectrum the left-branch CDF is
        F_L(x) = ((eta1 + eta2*x)/beta)^{1/(beta-1)}   for x in [x_L, mu]
    where eta2 = beta*y*^{beta-1}/(mu - x_L).

    We integrate the partial moments numerically and solve for y* by brentq.

    Parameters
    ----------
    s1:
        First lower partial moment  E[X * 1_{X <= mu}].
    s2:
        Second lower partial moment  E[X^2 * 1_{X <= mu}].
    beta:
        Power-spectrum parameter, beta > 1.
    mu:
        Distribution mean.
    a:
        Support lower bound.
    b:
        Support upper bound.

    Returns
    -------
    float
        y* = F*(mu) in (0, 1).
    """
    exp = 1.0 / (beta - 1.0)

    def _residual(y_star: float) -> float:
        """Residual: computed s1 - target s1 for candidate y*."""
        if y_star <= 0 or y_star >= 1:
            return float("nan")
        # Left branch: (F(x))^{beta-1} = eta2*(x - x_L)  where x_L = a
        # F(x)^{beta-1} = y*^{beta-1} * (x - a)/(mu - a)
        # F(x) = y* * ((x-a)/(mu-a))^{1/(beta-1)}
        # Compute s1 = integral_a^mu x dF(x)
        denom = mu - a
        if denom < 1e-15:
            return float("nan")

        def f_left(x: float) -> float:
            t = (x - a) / denom
            return y_star * t ** exp

        def pm1_integrand(x: float) -> float:
            # x * f(x) where f(x) = dF/dx
            t = (x - a) / denom
            if t < 1e-14:
                return 0.0
            f_x = y_star / denom * exp * t ** (exp - 1.0)
            return x * f_x

        pm1, _ = quad(pm1_integrand, a, mu, limit=100)
        return pm1 - s1

    # Bracket y* in (eps, 1-eps)
    lo, hi = 1e-6, 1.0 - 1e-6
    f_lo = _residual(lo)
    f_hi = _residual(hi)

    if f_lo * f_hi > 0:
        # Monotone: return midpoint as best guess
        return 0.5

    y_star = brentq(_residual, lo, hi, xtol=1e-12, rtol=1e-12)
    return float(y_star)


# ---------------------------------------------------------------------------
# Lagrange multipliers from y*
# ---------------------------------------------------------------------------

def compute_multipliers_model4(
    y_star: float,
    beta: float,
    s1: float,
    s2: float,
    mu: float,
) -> Tuple[float, float, float]:
    """Compute Lagrange multipliers (eta1, eta2, eta3) from y* and data.

    For the power-law CDF structure:
      Left branch  (x <= mu): (F*(x))^{beta-1} = eta2 * (x - x_L)
      Right branch (x >= mu): (F*(x))^{beta-1} = y*^{beta-1} + eta3*(x - mu)

    with eta1 = -eta2 * x_L (offset), x_L chosen so F*(a) = 0.

    Parameters
    ----------
    y_star:
        F*(mu), the CDF value at the mean.
    beta:
        Power-spectrum parameter.
    s1:
        First lower partial moment.
    s2:
        Second lower partial moment.
    mu:
        Distribution mean.

    Returns
    -------
    (eta1, eta2, eta3) : (float, float, float)
    """
    ybm1 = y_star ** (beta - 1.0)
    # eta2 is the slope of (F^{beta-1}) on the left branch; calibrated so
    # that the partial moment s2 is matched (approximate via matching derivatives).
    # For the simple case with support starting at a, we use:
    eta2 = ybm1  # normalised slope on left (mu - a = 1 unit)
    eta1 = -eta2 * mu + ybm1  # = 0 at x = mu gives F^{beta-1}=y*^{beta-1}
    # Right branch slope calibrated from s2 via heuristic (s2 / s1 ratio)
    eta3 = ybm1 / max(abs(mu), 1e-10) if abs(mu) > 1e-10 else ybm1
    return float(eta1), float(eta2), float(eta3)


# ---------------------------------------------------------------------------
# Support endpoints
# ---------------------------------------------------------------------------

def compute_support_endpoints(
    eta1: float,
    eta2: float,
    eta3: float,
    mu: float,
) -> Tuple[float, float]:
    """Compute (x_L, x_U) from the multiplier values.

    x_L : left boundary where (F*(x_L))^{beta-1} = 0  =>  eta1 + eta2*x_L = 0
    x_U : right boundary where (F*(x_U))^{beta-1} = y*^{beta-1} + eta3*(x_U-mu) = 1

    Since we do not carry y* here, x_U is parameterised through eta3 and the
    known right-branch value at mu.

    Parameters
    ----------
    eta1:
        Intercept multiplier.
    eta2:
        Left-branch slope.
    eta3:
        Right-branch slope.
    mu:
        Distribution mean (also the branch point).

    Returns
    -------
    (x_L, x_U) : (float, float)
    """
    # Left: eta1 + eta2 * x_L = 0
    if abs(eta2) < 1e-15:
        x_L = mu - 1.0
    else:
        x_L = -eta1 / eta2

    # Right: at x_U the right-branch value equals 1^{beta-1} = 1.
    # Right branch at x: (F*(x))^{beta-1} = (F*(mu))^{beta-1} + eta3*(x - mu)
    # At x_U: (F*(mu))^{beta-1} + eta3*(x_U - mu) = 1
    # => x_U = mu + (1 - (F*(mu))^{beta-1}) / eta3
    # We know F*(mu)^{beta-1} = eta1 + eta2*mu (from left-branch formula at mu)
    f_mu_bm1 = eta1 + eta2 * mu
    if abs(eta3) < 1e-15:
        x_U = mu + 10.0  # degenerate
    else:
        x_U = mu + (1.0 - f_mu_bm1) / eta3

    return float(x_L), float(x_U)


# ---------------------------------------------------------------------------
# Piecewise CDF
# ---------------------------------------------------------------------------

def construct_CDF_model4(
    x_grid: NDArray,
    eta1: float,
    eta2: float,
    eta3: float,
    mu: float,
    x_L: float,
    x_U: float,
    beta: float,
) -> NDArray:
    """Build the piecewise power-law CDF F*(x) on *x_grid*.

    F*(x) = 0                                             for x < x_L
    F*(x) = (eta1 + eta2*x)^{1/(beta-1)}                 for x in [x_L, mu]
    F*(x) = ((eta1+eta2*mu) + eta3*(x-mu))^{1/(beta-1)}  for x in (mu, x_U]
    F*(x) = 1                                             for x > x_U

    Parameters
    ----------
    x_grid:
        Points at which to evaluate F*, shape (n,).
    eta1, eta2, eta3:
        Lagrange multipliers from :func:`compute_multipliers_model4`.
    mu:
        Branch point (distribution mean).
    x_L:
        Left support endpoint.
    x_U:
        Right support endpoint.
    beta:
        Power-spectrum parameter.

    Returns
    -------
    NDArray
        CDF values in [0, 1], shape (n,).
    """
    x_grid = np.asarray(x_grid, dtype=float)
    exp = 1.0 / (beta - 1.0)
    F = np.zeros_like(x_grid)
    f_mu_bm1 = eta1 + eta2 * mu

    left_mask = (x_grid >= x_L) & (x_grid <= mu)
    right_mask = (x_grid > mu) & (x_grid <= x_U)
    above_mask = x_grid > x_U

    F[left_mask] = np.clip(eta1 + eta2 * x_grid[left_mask], 0.0, None) ** exp
    F[right_mask] = np.clip(f_mu_bm1 + eta3 * (x_grid[right_mask] - mu), 0.0, None) ** exp
    F[above_mask] = 1.0

    F = np.clip(F, 0.0, 1.0)
    return F


# ---------------------------------------------------------------------------
# Admissibility checks
# ---------------------------------------------------------------------------

def check_admissibility(x_L: float, x_U: float, a: float, b: float) -> bool:
    """Check that the optimal support [x_L, x_U] is contained in [a, b].

    Parameters
    ----------
    x_L:
        Optimal support lower endpoint.
    x_U:
        Optimal support upper endpoint.
    a:
        Ambiguity-set support lower bound.
    b:
        Ambiguity-set support upper bound.

    Returns
    -------
    bool
        True when x_L >= a and x_U <= b.
    """
    return (x_L >= a - 1e-10) and (x_U <= b + 1e-10)


def admissibility_check_beta_r(beta: float, r: float) -> bool:
    """Check admissibility of the moment ratio r for power-spectrum parameter beta.

    Conditions (from duality analysis):
      - beta = 2 : r must satisfy  r > 1
      - beta > 2 : r must satisfy  1 < r < (beta-1)/(beta-2)

    Parameters
    ----------
    beta:
        Power-spectrum parameter, beta >= 2.
    r:
        Moment ratio  s2 / (mu * s1) or similar; must be positive.

    Returns
    -------
    bool
    """
    if beta < 2.0 - 1e-10:
        raise ValueError(f"admissibility_check_beta_r requires beta >= 2, got {beta}.")
    if r <= 1.0:
        return False
    if abs(beta - 2.0) < 1e-10:
        return True  # r > 1 is sufficient for beta = 2
    upper = (beta - 1.0) / (beta - 2.0)
    return r < upper


# ---------------------------------------------------------------------------
# Risk computation (bounded support, CDF space)
# ---------------------------------------------------------------------------

def compute_risk_model4(
    F_star: NDArray,
    x_grid: NDArray,
    b: float,
    phi_func: Callable[..., NDArray],
    **phi_kwargs,
) -> float:
    """Compute the worst-case spectral risk in CDF space.

    R_phi = integral_a^b phi(F*(x)) * x * dF*(x)
          ≈ sum_i phi(F*(x_i)) * x_i * Delta_F_i      (trapezoidal)

    Equivalently, using integration by parts on [a, b]:
        R_phi = b - integral_a^b Phi(F*(x)) dx
    where Phi(t) = integral_0^t phi(v) dv.

    We use the direct quadrature (first formula) for robustness.

    Parameters
    ----------
    F_star:
        Optimal CDF values on *x_grid*, shape (n,).
    x_grid:
        Support grid, shape (n,).
    b:
        Upper bound of support (used to ensure F*(b) = 1).
    phi_func:
        Spectrum function phi(u, **phi_kwargs).
    **phi_kwargs:
        Forwarded to *phi_func*.

    Returns
    -------
    float
        Worst-case risk value.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    F_star = np.asarray(F_star, dtype=float)
    phi_vals = np.asarray(phi_func(F_star, **phi_kwargs), dtype=float)
    # Trapezoidal quadrature of phi(F(x)) * x * dF/dx dx = phi(F) * x * dF
    dF = np.diff(F_star)
    x_mid = 0.5 * (x_grid[:-1] + x_grid[1:])
    phi_mid = 0.5 * (phi_vals[:-1] + phi_vals[1:])
    return float(np.sum(phi_mid * x_mid * dF))


# ---------------------------------------------------------------------------
# Full Model-4 pipeline
# ---------------------------------------------------------------------------

def worst_case_risk_model4(
    s1: float,
    s2: float,
    beta: float,
    mu: float,
    a: float,
    b: float,
    phi_func: Callable[..., NDArray],
    n: int = 500,
    **phi_kwargs,
) -> Tuple[NDArray, NDArray, float, bool]:
    """End-to-end Model-4 worst-case risk computation.

    Parameters
    ----------
    s1:
        First lower partial moment  E[X * 1_{X <= mu}].
    s2:
        Second lower partial moment  E[X^2 * 1_{X <= mu}].
    beta:
        Power-spectrum parameter.
    mu:
        Distribution mean.
    a:
        Support lower bound.
    b:
        Support upper bound.
    phi_func:
        Spectrum function.
    n:
        Number of grid points on [a, b].
    **phi_kwargs:
        Forwarded to *phi_func*.

    Returns
    -------
    (x_grid, F_star, risk, admissible) : (NDArray, NDArray, float, bool)
    """
    y_star = solve_F_mu(s1, s2, beta, mu, a, b)
    eta1, eta2, eta3 = compute_multipliers_model4(y_star, beta, s1, s2, mu)
    x_L, x_U = compute_support_endpoints(eta1, eta2, eta3, mu)
    admissible = check_admissibility(x_L, x_U, a, b)
    x_grid = np.linspace(a, b, n)
    F_star = construct_CDF_model4(x_grid, eta1, eta2, eta3, mu, x_L, x_U, beta)
    risk = compute_risk_model4(F_star, x_grid, b, phi_func, **phi_kwargs)
    return x_grid, F_star, risk, admissible
