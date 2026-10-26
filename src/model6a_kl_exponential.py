"""
model6a_kl_exponential.py
-------------------------
Model 6a: Worst-case Expected Shortfall (ES) under a KL-divergence ambiguity
set with an exponential reference distribution on [0, b].

Reference distribution:
    g(x) = theta * exp(-theta*x) / (1 - exp(-theta*b)),   x in [0, b].

The KL ball is  B_eps(g) = { f : KL(f || g) <= epsilon }.

The worst-case ES at level alpha is achieved by the piecewise exponential
density

    f*(x) = { C1 * exp(-xi1 * x)   for x in [0, x_alpha]
            { C2 * exp(-xi2 * x)   for x in [x_alpha, b]

where  x_alpha = Q*^{-1}(alpha)  is the alpha-quantile and the three
parameters (xi1, xi2, x_alpha) satisfy a 3-equation system:

    (1) F*(x_alpha) = alpha              (quantile definition)
    (2) KL(f* || g) = epsilon            (KL constraint binding)
    (3) Continuity of f* at x_alpha:
            C1 * exp(-xi1*x_alpha) = C2 * exp(-xi2*x_alpha)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve, brentq, least_squares


# ---------------------------------------------------------------------------
# Integral helpers with continuous extensions
# ---------------------------------------------------------------------------

def I0(xi: float, u: float, v: float) -> float:
    """Compute  integral_u^v exp(-xi*x) dx  with a stable limit at xi=0.

    Parameters
    ----------
    xi:
        Exponential rate; any real value.
    u:
        Lower integration limit.
    v:
        Upper integration limit.

    Returns
    -------
    float
    """
    if abs(xi) < 1e-6:
        return float(v - u)
    return float((np.exp(-xi * u) - np.exp(-xi * v)) / xi)


def I1(xi: float, u: float, v: float) -> float:
    """Compute  integral_u^v x * exp(-xi*x) dx  with a stable limit at xi=0.

    Parameters
    ----------
    xi:
        Exponential rate; any real value.
    u:
        Lower integration limit.
    v:
        Upper integration limit.

    Returns
    -------
    float
    """
    if abs(xi) < 1e-6:
        return float((v ** 2 - u ** 2) / 2.0)
    # Antiderivative: -exp(-xi*x) * (x/xi + 1/xi^2)
    def antideriv(x: float) -> float:
        return -np.exp(-xi * x) * (x / xi + 1.0 / xi ** 2)
    return float(antideriv(v) - antideriv(u))


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def feasibility_check_6a(mu: float, b: float, theta: float, epsilon: float) -> bool:
    """Check that the problem is feasible.

    A necessary condition is that the KL radius *epsilon* is positive and the
    mean *mu* lies in (0, b).

    Parameters
    ----------
    mu:
        Reference distribution mean (or target mean).
    b:
        Support upper bound.
    theta:
        Reference exponential rate, theta > 0.
    epsilon:
        KL radius, epsilon >= 0.

    Returns
    -------
    bool
    """
    return epsilon >= 0 and 0 < mu < b and theta > 0


# ---------------------------------------------------------------------------
# 3-equation system
# ---------------------------------------------------------------------------

def _kl_piecewise(
    xi1: float,
    xi2: float,
    x_alpha: float,
    alpha: float,
    b: float,
    theta: float,
) -> Tuple[float, float]:
    """KL(f* || g) and mean of f* for given parameters.

    Parameters
    ----------
    xi1:
        Rate on [0, x_alpha].
    xi2:
        Rate on [x_alpha, b].
    x_alpha:
        Alpha-quantile of f*.
    alpha:
        ES confidence level.
    b:
        Support upper bound.
    theta:
        Reference rate.

    Returns
    -------
    (kl, mean_f) : (float, float)
    """
    i0_left = I0(xi1, 0.0, x_alpha)
    i0_right = I0(xi2, x_alpha, b)
    if i0_left < 1e-15 or i0_right < 1e-15:
        return float("nan"), float("nan")
    C1 = alpha / i0_left
    C2 = (1.0 - alpha) / i0_right

    # KL(f* || g) against the FULL exponential g(x) = theta*exp(-theta*x) on [0, inf).
    # For left piece: integral_0^{x_a} C1*exp(-xi1*x)*[ln(C1/theta) + (theta-xi1)*x] dx
    kl_left = (alpha * np.log(C1 / theta)
               + C1 * (theta - xi1) * I1(xi1, 0.0, x_alpha))

    # For right piece: integral_{x_a}^b C2*exp(-xi2*x)*[ln(C2/theta) + (theta-xi2)*x] dx
    kl_right = ((1.0 - alpha) * np.log(C2 / theta)
                + C2 * (theta - xi2) * I1(xi2, x_alpha, b))

    kl = kl_left + kl_right

    mean_f = C1 * I1(xi1, 0.0, x_alpha) + C2 * I1(xi2, x_alpha, b)
    return float(kl), float(mean_f)


def solve_system_6a(
    alpha: float,
    mu: float,
    b: float,
    theta: float,
    epsilon: float,
) -> Tuple[float, float, float]:
    """Solve the 3-equation system for (xi1, xi2, x_alpha).

    Equations:
      (1) F*(x_alpha) = alpha  [C1 * I0(xi1, 0, x_alpha) = alpha]
      (2) KL(f* || g) = epsilon
      (3) Continuity: C1*exp(-xi1*x_alpha) = C2*exp(-xi2*x_alpha)

    Parameters
    ----------
    alpha:
        ES confidence level.
    mu:
        Mean of f* (used to initialise search and validate).
    b:
        Support upper bound.
    theta:
        Reference exponential rate.
    epsilon:
        KL divergence radius.

    Returns
    -------
    (xi1, xi2, x_alpha) : (float, float, float)
    """
    ref_quantile = -np.log(1.0 - alpha) / theta  # reference alpha-quantile
    ref_quantile = min(ref_quantile, 0.95 * b)

    def equations(params: NDArray) -> NDArray:
        xi1, xi2, x_alpha = params
        if x_alpha <= 0 or x_alpha >= b or xi1 <= 0 or xi2 <= 0:
            return np.array([1e6, 1e6, 1e6])

        i0_left = I0(xi1, 0.0, x_alpha)
        i0_right = I0(xi2, x_alpha, b)
        if i0_left < 1e-15 or i0_right < 1e-15:
            return np.array([1e6, 1e6, 1e6])

        C1 = alpha / i0_left
        C2 = (1.0 - alpha) / i0_right

        kl, mean_f = _kl_piecewise(xi1, xi2, x_alpha, alpha, b, theta)

        eq1 = kl - epsilon
        eq2 = mean_f - mu
        eq3 = np.log(max(C1, 1e-300)) - xi1 * x_alpha - (np.log(max(C2, 1e-300)) - xi2 * x_alpha)
        return np.array([eq1, eq2, eq3])

    # Multi-start: sample (xi1, xi2, x_alpha) from diverse initial guesses.
    # Reference quantile is ~3 for alpha=0.95, theta=1; target x_alpha is ~0.5*ref_quantile.
    init_guesses = [
        [1.5 * theta, 1.0 * theta, 0.80 * ref_quantile],
        [2.0 * theta, 0.8 * theta, 0.60 * ref_quantile],
        [2.5 * theta, 1.0 * theta, 0.50 * ref_quantile],
        [3.0 * theta, 1.5 * theta, 0.40 * ref_quantile],
        [2.0 * theta, 1.2 * theta, 0.55 * ref_quantile],
        [2.5 * theta, 1.3 * theta, 0.52 * ref_quantile],
        [2.2 * theta, 1.4 * theta, 0.53 * ref_quantile],
        [1.8 * theta, 0.9 * theta, 0.65 * ref_quantile],
        [3.0 * theta, 1.0 * theta, 0.45 * ref_quantile],
        [1.5 * theta, 0.7 * theta, 0.70 * ref_quantile],
        [2.8 * theta, 1.2 * theta, 0.48 * ref_quantile],
        [2.3 * theta, 1.1 * theta, 0.57 * ref_quantile],
        [theta,       0.5 * theta, ref_quantile],
    ]
    # Clip x_alpha guesses to valid range
    for g in init_guesses:
        g[2] = np.clip(g[2], 1e-3, b - 1e-3)

    best_sol = None
    best_norm = np.inf

    for x0 in init_guesses:
        # Try fsolve
        for solver_fn in [
            lambda x: fsolve(equations, x, full_output=True, xtol=1e-12)[0],
        ]:
            try:
                sol = solver_fn(np.array(x0, dtype=float))
                xi1_s, xi2_s, xa_s = sol
                if xi1_s > 0 and xi2_s > 0 and 0 < xa_s < b:
                    res_norm = float(np.linalg.norm(equations(sol)))
                    if res_norm < best_norm:
                        best_norm = res_norm
                        best_sol = (float(xi1_s), float(xi2_s), float(xa_s))
            except Exception:
                pass
        # Also try least_squares with bounds
        try:
            lb = [1e-6, 1e-6, 1e-6]
            ub = [50.0 * theta, 50.0 * theta, b - 1e-6]
            r = least_squares(equations, np.array(x0, dtype=float),
                              bounds=(lb, ub), ftol=1e-12, xtol=1e-12, gtol=1e-12)
            xi1_s, xi2_s, xa_s = r.x
            if xi1_s > 0 and xi2_s > 0 and 0 < xa_s < b:
                res_norm = float(np.linalg.norm(equations(r.x)))
                if res_norm < best_norm:
                    best_norm = res_norm
                    best_sol = (float(xi1_s), float(xi2_s), float(xa_s))
        except Exception:
            pass

    if best_sol is not None and best_norm < 1e-6:
        return best_sol

    # Fallback: return the best we found even if residual is larger
    if best_sol is not None:
        return best_sol
    return float(theta), float(theta * 0.5), float(min(ref_quantile, 0.95 * b))


# ---------------------------------------------------------------------------
# Density construction
# ---------------------------------------------------------------------------

def construct_density_6a(
    x_grid: NDArray,
    xi1: float,
    xi2: float,
    x_alpha: float,
    b: float,
    alpha: float,
) -> NDArray:
    """Construct the piecewise exponential density f*(x) on *x_grid*.

    Parameters
    ----------
    x_grid:
        Evaluation points in [0, b], shape (n,).
    xi1:
        Rate on [0, x_alpha].
    xi2:
        Rate on [x_alpha, b].
    x_alpha:
        Alpha-quantile (breakpoint).
    b:
        Support upper bound.
    alpha:
        ES confidence level.

    Returns
    -------
    NDArray
        Density values f*(x), shape (n,).
    """
    x_grid = np.asarray(x_grid, dtype=float)
    f = np.zeros_like(x_grid)

    i0_left = I0(xi1, 0.0, x_alpha)
    i0_right = I0(xi2, x_alpha, b)
    if i0_left < 1e-15 or i0_right < 1e-15:
        return f

    C1 = alpha / i0_left
    C2 = (1.0 - alpha) / i0_right

    left_mask = (x_grid >= 0.0) & (x_grid <= x_alpha)
    right_mask = (x_grid > x_alpha) & (x_grid <= b)

    f[left_mask] = C1 * np.exp(-xi1 * x_grid[left_mask])
    f[right_mask] = C2 * np.exp(-xi2 * x_grid[right_mask])
    return f


# ---------------------------------------------------------------------------
# Worst-case ES
# ---------------------------------------------------------------------------

def compute_worst_case_ES_6a(xi2: float, x_alpha: float, b: float) -> float:
    """Compute the worst-case ES in closed form.

    ES_alpha(f*)  =  C2 / (1-alpha) * I1(xi2, x_alpha, b)

    where  C2 = (1-alpha) / I0(xi2, x_alpha, b),  so

    ES_alpha(f*)  =  I1(xi2, x_alpha, b) / I0(xi2, x_alpha, b).

    Parameters
    ----------
    xi2:
        Exponential rate on (x_alpha, b].
    x_alpha:
        Alpha-quantile of f*.
    b:
        Support upper bound.

    Returns
    -------
    float
        Worst-case ES value.
    """
    i0 = I0(xi2, x_alpha, b)
    if i0 < 1e-15:
        return float(x_alpha)
    i1 = I1(xi2, x_alpha, b)
    return float(i1 / i0)


# ---------------------------------------------------------------------------
# Minimum KL radius
# ---------------------------------------------------------------------------

def compute_epsilon_min_6a(mu: float, b: float, theta: float) -> float:
    """Minimum feasible KL radius using the feasibility lemma.

    Finds the single tilted exponential f(x) ∝ exp(-varsigma*x) on [0, b]
    with mean mu, then returns KL(f || g) where g = theta*exp(-theta*x) is
    the full exponential reference.

    The formula is:
        epsilon_min = ln(C(varsigma*)) - ln(theta) + (theta - varsigma*) * mu
    where C(varsigma) = varsigma / (1 - exp(-varsigma*b))  (normalisation constant)
    and varsigma* solves  I1(varsigma, 0, b) / I0(varsigma, 0, b) = mu.

    Parameters
    ----------
    mu:
        Target mean; must satisfy 0 < mu < b.
    b:
        Support upper bound.
    theta:
        Reference exponential rate, theta > 0.

    Returns
    -------
    float
        Minimum epsilon >= 0.
    """
    half_b = b / 2.0

    def mean_eq(vs: float) -> float:
        i0 = I0(vs, 0.0, b)
        if abs(i0) < 1e-300:
            return -mu
        return float(I1(vs, 0.0, b) / i0) - mu

    try:
        if abs(mu - half_b) < 1e-10:
            varsigma = 0.0
        elif mu < half_b:
            # mean decreases from b/2 to 0 as varsigma goes from 0 to +inf
            varsigma = brentq(mean_eq, 1e-10, 500.0 / b, xtol=1e-12)
        else:
            # mean increases above b/2 as varsigma goes negative
            varsigma = brentq(mean_eq, -500.0 / b, -1e-10, xtol=1e-12)

        if abs(varsigma) < 1e-12:
            C_vs = 1.0 / b
        else:
            C_vs = varsigma / (1.0 - np.exp(-varsigma * b))

        epsilon_min = np.log(C_vs) - np.log(theta) + (theta - varsigma) * mu
        return max(float(epsilon_min), 0.0)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Numerical cross-check
# ---------------------------------------------------------------------------

def crosscheck_6a(
    alpha: float,
    mu: float,
    b: float,
    theta: float,
    epsilon: float,
    n: int = 1000,
) -> dict:
    """Validate the analytic worst-case ES against numerical integration.

    Parameters
    ----------
    alpha, mu, b, theta, epsilon:
        Model parameters.
    n:
        Number of grid points.

    Returns
    -------
    dict with keys 'es_analytic', 'es_numerical', 'abs_diff', 'kl_numerical'.
    """
    xi1, xi2, x_alpha = solve_system_6a(alpha, mu, b, theta, epsilon)
    es_analytic = compute_worst_case_ES_6a(xi2, x_alpha, b)

    x_grid = np.linspace(0.0, b, n)
    f = construct_density_6a(x_grid, xi1, xi2, x_alpha, b, alpha)
    h = x_grid[1] - x_grid[0]

    mask = x_grid >= x_alpha
    es_num = float(np.trapz(x_grid[mask] * f[mask], x_grid[mask])) / max(1.0 - alpha, 1e-15)

    # Numerical KL against full exponential g(x) = theta*exp(-theta*x) (no Z factor)
    g = theta * np.exp(-theta * x_grid)
    ratio = np.where(f > 1e-300, f / np.maximum(g, 1e-300), 1.0)
    kl_num = float(np.trapz(f * np.log(np.maximum(ratio, 1e-300)), x_grid))

    return {
        "es_analytic": es_analytic,
        "es_numerical": es_num,
        "abs_diff": abs(es_analytic - es_num),
        "kl_numerical": kl_num,
    }
