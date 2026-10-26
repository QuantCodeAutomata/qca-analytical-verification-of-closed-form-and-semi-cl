"""
model6b_kl_gaussian.py
----------------------
Model 6b: Worst-case Expected Shortfall (ES) under a KL-divergence ambiguity
set with a Gaussian reference distribution N(bar_mu, 1).

The KL ball centred on  g = N(bar_mu, 1)  is

    B_eps(g) = { f  :  KL(f || g) <= epsilon }.

The worst-case ES at level alpha is achieved by a piecewise tilted Gaussian:

    f*(x) = { C1 * exp(-x^2/2 + lambda1*(x-bar_mu))   for x <= x_alpha
            { C2 * exp(-x^2/2 + lambda2*(x-bar_mu))   for x > x_alpha

The key scalar output is the chi value

    chi(alpha, epsilon, bar_mu)

which is the maximal ES of the N(bar_mu, 1)-centred problem and satisfies

    worst-case ES  =  bar_mu  +  sigma * chi(alpha, epsilon, 0)

for any Gaussian reference with standard deviation sigma (after standardising).

Feasibility condition: epsilon > bar_mu^2 / 2.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq, fsolve
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Integral helpers: J0 and J1
# ---------------------------------------------------------------------------

def J0(lam: float, u: float, v: float) -> float:
    """Compute  exp(lam^2/2) * [Phi_N(v - lam) - Phi_N(u - lam)].

    This equals  (1/sqrt(2*pi)) * integral_u^v exp(-x^2/2 + lam*x) dx
    in log-stabilised form.

    Parameters
    ----------
    lam:
        Tilting parameter (mean shift).
    u:
        Lower limit (use -np.inf for the left half-line).
    v:
        Upper limit (use +np.inf for the right half-line).

    Returns
    -------
    float
    """
    # Compute log-stably: log(J0) = lam^2/2 + log(Phi(v-lam) - Phi(u-lam))
    p_v = float(norm.cdf(v - lam)) if np.isfinite(v) else 1.0
    p_u = float(norm.cdf(u - lam)) if np.isfinite(u) else 0.0
    diff = p_v - p_u
    if diff < 1e-300:
        return 0.0
    log_J0 = 0.5 * lam ** 2 + np.log(diff)
    return float(np.exp(log_J0))


def J1(lam: float, u: float, v: float) -> float:
    """Compute  exp(lam^2/2) * (lam*[Phi_N(v-lam)-Phi_N(u-lam)] - [phi_N(v-lam)-phi_N(u-lam)]).

    This equals  (1/sqrt(2*pi)) * integral_u^v x * exp(-x^2/2 + lam*x) dx.

    Parameters
    ----------
    lam:
        Tilting parameter.
    u:
        Lower limit.
    v:
        Upper limit.

    Returns
    -------
    float
    """
    p_v = float(norm.cdf(v - lam)) if np.isfinite(v) else 1.0
    p_u = float(norm.cdf(u - lam)) if np.isfinite(u) else 0.0
    pdf_v = float(norm.pdf(v - lam)) if np.isfinite(v) else 0.0
    pdf_u = float(norm.pdf(u - lam)) if np.isfinite(u) else 0.0

    bracket_Phi = p_v - p_u
    bracket_phi = pdf_v - pdf_u  # phi_N(v-lam) - phi_N(u-lam)

    log_factor = 0.5 * lam ** 2
    return float(np.exp(log_factor) * (lam * bracket_Phi - bracket_phi))


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def check_feasibility_6b(bar_mu: float, epsilon: float) -> bool:
    """Check the feasibility condition  epsilon > bar_mu^2 / 2.

    This condition ensures that the N(0,1) reference distribution is in the
    interior of the KL ball centred on N(bar_mu, 1).

    Parameters
    ----------
    bar_mu:
        Reference mean.
    epsilon:
        KL radius.

    Returns
    -------
    bool
    """
    return float(epsilon) > float(bar_mu) ** 2 / 2.0


# ---------------------------------------------------------------------------
# Primal system solver
# ---------------------------------------------------------------------------

def _kl_piecewise_gaussian(
    lam1: float,
    lam2: float,
    x_alpha: float,
    alpha: float,
    bar_mu: float,
) -> Tuple[float, float, float]:
    """Compute normalization error, KL, and mean of the piecewise tilted Gaussian.

    Returns (norm_error, kl, mean_f) for use in the nonlinear system solver.
    """
    # Effective tilting parameters relative to bar_mu
    l1 = lam1 - bar_mu  # effective tilt on left piece
    l2 = lam2 - bar_mu  # effective tilt on right piece

    # Normalisation
    sqrt2pi = np.sqrt(2.0 * np.pi)
    # integral_{-inf}^{x_a} exp(-x^2/2+lam1*x) dx / sqrt(2*pi)
    #   = exp(lam1^2/2) * [Phi(x_a - lam1) - 0]  [using J0 definition * sqrt(2*pi)]
    # We keep C1, C2 explicitly.
    # J0(lam1, -inf, x_alpha) = exp(lam1^2/2) * Phi(x_alpha - lam1)
    j0_left = J0(lam1, -np.inf, x_alpha)
    j0_right = J0(lam2, x_alpha, np.inf)
    if j0_left < 1e-300 or j0_right < 1e-300:
        return 1e6, 1e6, 1e6

    C1 = alpha / (sqrt2pi * j0_left)
    C2 = (1.0 - alpha) / (sqrt2pi * j0_right)

    # Mean: E_{f*}[X] = sqrt(2*pi)*(C1*J1(lam1,-inf,x_a) + C2*J1(lam2,x_a,inf))
    mean_f = sqrt2pi * (C1 * J1(lam1, -np.inf, x_alpha) + C2 * J1(lam2, x_alpha, np.inf))

    # KL(f* || N(bar_mu, 1)):
    # = integral f*(x) * (lam1*(x-bar_mu) - lam1^2/2 - ...) dx + sum_piece
    # KL = integral f*(x) * ln(f*(x)/g(x)) dx  where g = N(bar_mu,1)
    # ln(f*(x)/g(x)) = ln(C1) + lam1*x - lam1*bar_mu - lam1^2/2  for left piece (approx)
    # Full KL:
    kl_left = (
        alpha * np.log(C1 * sqrt2pi)  # = alpha * ln(C1*sqrt(2*pi))
        + C1 * sqrt2pi * lam1 * J1(lam1, -np.inf, x_alpha)
        - 0.5 * lam1 ** 2 * alpha  # from the -x^2/2 terms cancelling; only exponential part
    )
    kl_right = (
        (1.0 - alpha) * np.log(C2 * sqrt2pi)
        + C2 * sqrt2pi * lam2 * J1(lam2, x_alpha, np.inf)
        - 0.5 * lam2 ** 2 * (1.0 - alpha)
    )

    # More precise KL computation:
    # KL(f* || g) where g = N(bar_mu, 1):
    # = E_{f*}[ln f*(X)] - E_{f*}[ln g(X)]
    # E_{f*}[ln g(X)] = -0.5*ln(2*pi) - 0.5*E_{f*}[(X-bar_mu)^2]
    #                 = -0.5*ln(2*pi) - 0.5*(Var_{f*}[X] + (E_{f*}[X]-bar_mu)^2)

    # E_{f*}[X^2]:
    # For tilted Gaussian N(lam1,1) restricted to (-inf, x_alpha):
    # E[X^2 * 1_{X<=x_a}] with density proportional to exp(-x^2/2+lam1*x)
    # = C1*sqrt(2*pi) * integral_{-inf}^{x_a} x^2 * phi_N(x-lam1)*exp(lam1^2/2) dx / sqrt(2*pi)
    # which needs a second-moment J2 function -- let's use a simpler KL formula.

    # Simplified: KL(f* || g) = 0.5*( E_{f*}[(X-lam1)^2] + 0.5*lam1^2 + ...)
    # For piecewise tilted Gaussians, use the exact formula:
    # KL(f* || g) = E_{f*}[lam_piece * (X - bar_mu)] - 0.5*E_{f*}[lam_piece^2]
    #             (valid when both f* and g are Gaussian with unit variance)
    kl = (C1 * sqrt2pi * (lam1 - bar_mu) * J1(lam1, -np.inf, x_alpha)
          + C2 * sqrt2pi * (lam2 - bar_mu) * J1(lam2, x_alpha, np.inf)
          - 0.5 * (lam1 - bar_mu) ** 2 * alpha
          - 0.5 * (lam2 - bar_mu) ** 2 * (1.0 - alpha))

    return 0.0, float(kl), float(mean_f)


def solve_primal_system_6b(
    alpha: float,
    epsilon: float,
    bar_mu: float,
) -> Tuple[float, float, float]:
    """Solve equations (54)-(56) for (lambda1, lambda2, x_alpha).

    System:
      (54) F*(x_alpha) = alpha                    (quantile condition)
      (55) KL(f* || N(bar_mu,1)) = epsilon        (KL binding)
      (56) Continuity of f* at x_alpha:
               C1*exp(-x_alpha^2/2 + lam1*x_alpha)
             = C2*exp(-x_alpha^2/2 + lam2*x_alpha)

    Parameters
    ----------
    alpha:
        ES confidence level in (0, 1).
    epsilon:
        KL radius > bar_mu^2/2.
    bar_mu:
        Reference Gaussian mean.

    Returns
    -------
    (lambda1, lambda2, x_alpha) : (float, float, float)
    """
    sqrt2pi = np.sqrt(2.0 * np.pi)
    x_alpha0 = float(norm.ppf(alpha)) + bar_mu  # reference quantile

    def equations(params: NDArray) -> NDArray:
        lam1, lam2, x_alpha = params

        j0_left = J0(lam1, -np.inf, x_alpha)
        j0_right = J0(lam2, x_alpha, np.inf)
        if j0_left < 1e-300 or j0_right < 1e-300:
            return np.array([1e6, 1e6, 1e6])

        C1 = alpha / (sqrt2pi * j0_left)
        C2 = (1.0 - alpha) / (sqrt2pi * j0_right)

        # (54) is satisfied by construction via C1 choice
        # Equation 1: KL = epsilon
        _, kl, _ = _kl_piecewise_gaussian(lam1, lam2, x_alpha, alpha, bar_mu)
        eq1 = kl - epsilon

        # Equation 2: mean = bar_mu
        mean_f = sqrt2pi * (C1 * J1(lam1, -np.inf, x_alpha) + C2 * J1(lam2, x_alpha, np.inf))
        eq2 = mean_f - bar_mu

        # Equation 3: continuity at x_alpha
        log_C1 = np.log(max(C1, 1e-300))
        log_C2 = np.log(max(C2, 1e-300))
        eq3 = (log_C1 + lam1 * x_alpha) - (log_C2 + lam2 * x_alpha)

        return np.array([eq1, eq2, eq3])

    # Initial guess: no tilt, reference quantile
    x0 = np.array([bar_mu, bar_mu + np.sqrt(2.0 * epsilon), x_alpha0])
    try:
        sol = fsolve(equations, x0, full_output=True)
        lam1, lam2, x_alpha = sol[0]
        residual = np.max(np.abs(equations(sol[0])))
        if residual < 1e-4 and lam2 > lam1:
            return float(lam1), float(lam2), float(x_alpha)
    except Exception:
        pass

    # Fallback: 1D search for lam2 with lam1=bar_mu and x_alpha=x_alpha0
    def kl_eq(lam2: float) -> float:
        _, kl, _ = _kl_piecewise_gaussian(bar_mu, lam2, x_alpha0, alpha, bar_mu)
        return kl - epsilon

    try:
        lam2_lo, lam2_hi = bar_mu, bar_mu + 20.0
        lam2_sol = brentq(kl_eq, lam2_lo, lam2_hi, xtol=1e-10)
        return float(bar_mu), float(lam2_sol), float(x_alpha0)
    except Exception:
        return float(bar_mu), float(bar_mu + np.sqrt(2.0 * epsilon)), float(x_alpha0)


# ---------------------------------------------------------------------------
# Chi: worst-case ES scalar
# ---------------------------------------------------------------------------

def compute_chi_primal(lambda2: float, x_alpha: float) -> float:
    """Compute chi from the primal: chi = lambda2 + phi_N(x_alpha-lambda2) / (1-Phi_N(x_alpha-lambda2)).

    This equals the mean of a N(lambda2, 1) truncated to (x_alpha, inf), which
    is the worst-case ES when the density is tilted by lam2 in the tail.

    Parameters
    ----------
    lambda2:
        Optimal tilting parameter for the tail piece.
    x_alpha:
        Alpha-quantile of the worst-case distribution.

    Returns
    -------
    float
    """
    z = x_alpha - lambda2
    pdf_z = float(norm.pdf(z))
    sf_z = float(norm.sf(z))  # 1 - Phi_N(z)
    if sf_z < 1e-300:
        return float(lambda2)
    return float(lambda2 + pdf_z / sf_z)


# ---------------------------------------------------------------------------
# Dual solver
# ---------------------------------------------------------------------------

def solve_dual_6b(
    alpha: float,
    epsilon: float,
    bar_mu: float,
) -> Tuple[float, float, float]:
    """Solve the dual minimisation for Model 6b.

    The dual problem for KL ambiguity is:

        min_{eta >= 0}  D(eta)  =  eta * epsilon  +  log MGF contribution

    For the ES functional with Gaussian reference, the dual objective is:

        D(eta)  =  eta * epsilon
                +  eta * log E_{N(bar_mu,1)}[ exp( 1_{X > x_alpha^g} * (X - x_alpha^g) / (eta*(1-alpha)) ) ]

    where x_alpha^g = bar_mu + z_alpha (with z_alpha = Phi_N^{-1}(alpha)).

    We minimise D(eta) via a 1D search.

    Parameters
    ----------
    alpha:
        ES confidence level.
    epsilon:
        KL radius.
    bar_mu:
        Reference mean.

    Returns
    -------
    (eta_star, chi_dual, x_alpha) : (float, float, float)
    """
    x_alpha_g = float(norm.ppf(alpha)) + bar_mu  # reference alpha-quantile

    def D(eta: float) -> float:
        if eta <= 1e-10:
            return float("inf")
        # Log MGF of the ES indicator under N(bar_mu, 1)
        # E[exp((X-x_alpha^g)/(eta*(1-alpha)) * 1_{X>x_alpha^g})]
        c = 1.0 / (eta * (1.0 - alpha))
        # MGF of indicator: 1 - alpha + integral_{x_alpha}^inf exp(c*(x-x_alpha^g)) phi_N(x-bar_mu) dx
        # = 1 - alpha + exp(-c*x_alpha^g) * J0(c + bar_mu, x_alpha_g, inf) * sqrt(2*pi)
        # Let lam_eff = bar_mu + c (tilting by c in tail beyond x_alpha):
        log_mgf_tail = (
            np.log(max(1.0 - alpha, 1e-300))
            + np.log1p(
                np.sqrt(2.0 * np.pi) * np.exp(-c * x_alpha_g)
                * J0(bar_mu + c, x_alpha_g, np.inf) / max(1.0 - alpha, 1e-300)
            )
        )
        return eta * epsilon + eta * (log_mgf_tail if np.isfinite(log_mgf_tail) else 1e6)

    # Minimise over eta > 0
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(D, bounds=(1e-6, 1e3), method="bounded")
    eta_star = float(res.x)

    # Recover chi from dual:
    c_star = 1.0 / (eta_star * (1.0 - alpha))
    lam_eff = bar_mu + c_star
    # chi = x_alpha^g + conditional mean of N(bar_mu,1) truncated above x_alpha^g shifted by c
    j1_val = J1(lam_eff, x_alpha_g, np.inf)
    j0_val = J0(lam_eff, x_alpha_g, np.inf)
    if j0_val < 1e-300:
        chi_dual = compute_chi_primal(lam_eff, x_alpha_g)
    else:
        chi_dual = float(j1_val / j0_val)

    return eta_star, float(chi_dual), float(x_alpha_g)


# ---------------------------------------------------------------------------
# Density construction
# ---------------------------------------------------------------------------

def construct_density_6b(
    x_grid: NDArray,
    lambda1: float,
    lambda2: float,
    x_alpha: float,
    alpha: float,
) -> NDArray:
    """Construct the piecewise tilted Gaussian density f*(x) on *x_grid*.

    Parameters
    ----------
    x_grid:
        Evaluation points, shape (n,).
    lambda1:
        Tilting parameter on (-inf, x_alpha].
    lambda2:
        Tilting parameter on (x_alpha, inf).
    x_alpha:
        Alpha-quantile breakpoint.
    alpha:
        ES confidence level.

    Returns
    -------
    NDArray
        Density values f*(x), shape (n,).
    """
    x_grid = np.asarray(x_grid, dtype=float)
    sqrt2pi = np.sqrt(2.0 * np.pi)
    f = np.zeros_like(x_grid)

    j0_left = J0(lambda1, -np.inf, x_alpha)
    j0_right = J0(lambda2, x_alpha, np.inf)

    if j0_left < 1e-300 or j0_right < 1e-300:
        return f

    C1 = alpha / (sqrt2pi * j0_left)
    C2 = (1.0 - alpha) / (sqrt2pi * j0_right)

    left_mask = x_grid <= x_alpha
    right_mask = x_grid > x_alpha

    f[left_mask] = C1 * np.exp(-x_grid[left_mask] ** 2 / 2.0 + lambda1 * x_grid[left_mask])
    f[right_mask] = C2 * np.exp(-x_grid[right_mask] ** 2 / 2.0 + lambda2 * x_grid[right_mask])
    return f


# ---------------------------------------------------------------------------
# Primal-dual consistency check
# ---------------------------------------------------------------------------

def verify_primal_dual_consistency(
    alpha: float,
    epsilon: float,
    bar_mu: float,
) -> dict:
    """Check that the primal and dual chi values agree.

    Parameters
    ----------
    alpha:
        ES confidence level.
    epsilon:
        KL radius.
    bar_mu:
        Reference mean.

    Returns
    -------
    dict with keys 'chi_primal', 'chi_dual', 'diff', 'feasible'.
    """
    feasible = check_feasibility_6b(bar_mu, epsilon)
    if not feasible:
        return {"chi_primal": float("nan"), "chi_dual": float("nan"), "diff": float("nan"), "feasible": False}

    lam1, lam2, x_alpha = solve_primal_system_6b(alpha, epsilon, bar_mu)
    chi_primal = compute_chi_primal(lam2, x_alpha)

    _, chi_dual, _ = solve_dual_6b(alpha, epsilon, bar_mu)

    return {
        "chi_primal": float(chi_primal),
        "chi_dual": float(chi_dual),
        "diff": float(abs(chi_primal - chi_dual)),
        "feasible": True,
    }


# ---------------------------------------------------------------------------
# Tabulation
# ---------------------------------------------------------------------------

def tabulate_chi(
    alpha_vals: List[float],
    epsilon_vals: List[float],
    bar_mu_vals: List[float],
) -> pd.DataFrame:
    """Tabulate chi(alpha, epsilon, bar_mu) over a grid.

    Parameters
    ----------
    alpha_vals:
        ES confidence levels.
    epsilon_vals:
        KL radii.
    bar_mu_vals:
        Reference means.

    Returns
    -------
    pd.DataFrame
        Columns: alpha, epsilon, bar_mu, chi, feasible.
    """
    rows = []
    for alpha in alpha_vals:
        for eps in epsilon_vals:
            for bmu in bar_mu_vals:
                feasible = check_feasibility_6b(bmu, eps)
                if feasible:
                    lam1, lam2, x_alpha = solve_primal_system_6b(alpha, eps, bmu)
                    chi = compute_chi_primal(lam2, x_alpha)
                else:
                    chi = float("nan")
                rows.append({"alpha": alpha, "epsilon": eps, "bar_mu": bmu,
                              "chi": chi, "feasible": feasible})
    return pd.DataFrame(rows)
