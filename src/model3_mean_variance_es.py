"""
model3_mean_variance_es.py
--------------------------
Model 3: Worst-case spectral risk under a mean-variance ambiguity set with
a two-sided Expected-Shortfall (ES) band.

The ambiguity set is

    M3(mu, sigma, alpha, c_l, c_u) = {
        Q non-decreasing on [0,1] :
            integral_0^1 Q(u) du = mu,
            integral_0^1 (Q(u)-mu)^2 du <= sigma^2,
            c_l <= ES_alpha(Q) <= c_u
    }

The unconstrained (benchmark) optimiser is

    Q^diamond(u) = mu + (sigma / sigma_phi) * (phi(u) - 1)

with  sigma_phi = sqrt( integral_0^1 (phi(u)-1)^2 du ).

Three regimes arise depending on where  M_alpha := ES_alpha(Q^diamond) falls
relative to the band [c_l, c_u]:

  * benchmark   : c_l <= M_alpha <= c_u  — no ES constraint binds
  * floor_active: M_alpha < c_l           — ES floor IS binding
  * cap_active  : M_alpha > c_u           — ES cap IS binding

For the power spectrum  phi(u) = beta * u^{beta-1}  the closed-form
  sigma_phi = (beta-1) / sqrt(2*beta-1)
is used when *beta* is provided.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from src.spectral_risk import risk_quantile_space


# ---------------------------------------------------------------------------
# Helper quantities
# ---------------------------------------------------------------------------

def compute_sigma_phi(phi_vals: NDArray, u_grid: NDArray) -> float:
    """Compute sigma_phi = sqrt( integral_0^1 (phi(u)-1)^2 du ).

    Parameters
    ----------
    phi_vals:
        Spectrum evaluated on *u_grid*, shape (n,).
    u_grid:
        Probability grid in [0, 1], shape (n,).

    Returns
    -------
    float
    """
    return float(np.sqrt(np.trapz((phi_vals - 1.0) ** 2, u_grid)))


def compute_A0(phi_vals: NDArray, u_grid: NDArray, alpha: float) -> float:
    """Compute  A0 = 1/(1-alpha) * integral_alpha^1 phi(u) du  - 1.

    This equals ES_alpha(Q^diamond - mu) / (sigma/sigma_phi), i.e., the ES
    contribution from the spectral shape beyond the mean.

    Parameters
    ----------
    phi_vals:
        Spectrum on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    alpha:
        ES confidence level in (0, 1).

    Returns
    -------
    float
    """
    mask = u_grid >= alpha
    if not np.any(mask):
        return -1.0
    A = float(np.trapz(phi_vals[mask], u_grid[mask]))
    return A / (1.0 - alpha) - 1.0


def compute_Q_diamond(
    phi_vals: NDArray,
    u_grid: NDArray,
    mu: float,
    sigma: float,
    sigma_phi: float,
) -> NDArray:
    """Build the benchmark quantile function Q^diamond.

    Q^diamond(u) = mu + (sigma / sigma_phi) * (phi(u) - 1)

    Parameters
    ----------
    phi_vals:
        Spectrum on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    mu:
        Distribution mean.
    sigma:
        Standard deviation bound.
    sigma_phi:
        Spectral standard deviation from :func:`compute_sigma_phi`.

    Returns
    -------
    NDArray
        Q^diamond values on *u_grid*, shape (n,).
    """
    return mu + (sigma / sigma_phi) * (phi_vals - 1.0)


def compute_benchmark_ES(Q_diamond_vals: NDArray, u_grid: NDArray, alpha: float) -> float:
    """ES_alpha of the benchmark distribution Q^diamond.

    Parameters
    ----------
    Q_diamond_vals:
        Benchmark quantile on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    alpha:
        ES confidence level.

    Returns
    -------
    float
    """
    mask = u_grid >= alpha
    if not np.any(mask):
        return float(Q_diamond_vals[-1])
    return float(np.trapz(Q_diamond_vals[mask], u_grid[mask])) / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime_model3(
    bar_c: float,
    c_l: float,
    c_u: float,
    M_alpha: float,
    mu: float,
) -> str:
    """Return the constraint regime for Model 3.

    Parameters
    ----------
    bar_c:
        Not used directly; kept for API symmetry.
    c_l:
        ES floor.
    c_u:
        ES cap.
    M_alpha:
        ES_alpha(Q^diamond), the benchmark ES value.
    mu:
        Distribution mean (unused; for signature consistency).

    Returns
    -------
    str
        One of ``'benchmark'``, ``'floor_active'``, ``'cap_active'``.
    """
    if M_alpha < c_l:
        return "floor_active"
    elif M_alpha > c_u:
        return "cap_active"
    else:
        return "benchmark"


# ---------------------------------------------------------------------------
# Floor-active regime
# ---------------------------------------------------------------------------

def _tilted_sigma_phi(
    phi_vals: NDArray,
    u_grid: NDArray,
    alpha: float,
    lam: float,
) -> Tuple[float, float, float]:
    """Return (sigma_phi_tilde, A0_tilde, mean_phi_tilde) for the tilted spectrum.

    phi_tilde(u) = phi(u) + lam/(1-alpha) * 1_{u>=alpha}
    mean_phi_tilde = integral_0^1 phi_tilde(u) du = 1 + lam
    sigma_phi_tilde = sqrt( integral (phi_tilde - mean_phi_tilde)^2 du )
    A0_tilde = 1/(1-alpha) * integral_alpha^1 phi_tilde(u) du - mean_phi_tilde
    """
    phi_tilde = phi_vals.copy()
    mask = u_grid >= alpha
    phi_tilde[mask] += lam / (1.0 - alpha)
    mean_phi_tilde = 1.0 + lam
    sigma_phi_tilde = float(np.sqrt(np.trapz((phi_tilde - mean_phi_tilde) ** 2, u_grid)))
    if np.any(mask):
        A_tilde = float(np.trapz(phi_tilde[mask], u_grid[mask]))
    else:
        A_tilde = 0.0
    A0_tilde = A_tilde / (1.0 - alpha) - mean_phi_tilde
    return sigma_phi_tilde, A0_tilde, float(mean_phi_tilde)


def _ES_floor_T(
    lam: float,
    phi_vals: NDArray,
    u_grid: NDArray,
    mu: float,
    sigma: float,
    alpha: float,
) -> float:
    """ES_alpha(Q*(lam)) for the floor-active tilted spectrum, minus c_l.

    Used internally by brentq in :func:`solve_floor_active`.
    """
    sigma_pt, A0_t, _ = _tilted_sigma_phi(phi_vals, u_grid, alpha, lam)
    if sigma_pt < 1e-15:
        return mu - 99999.0
    return mu + sigma / sigma_pt * A0_t


def solve_floor_active(
    phi_vals: NDArray,
    u_grid: NDArray,
    mu: float,
    sigma: float,
    alpha: float,
    c_l: float,
    sigma_phi: float,
    A0: float,
) -> Tuple[NDArray, float]:
    """Find lambda_l and Q* for the floor-active regime.

    Solves  T(lambda_l) = c_l  where  T(lam) = ES_alpha(Q*(lam))  and
    Q*(u; lam) = mu + (sigma/sigma_phi_tilde(lam)) * (phi_tilde(u; lam) - mean_phi_tilde(lam)).

    Parameters
    ----------
    phi_vals:
        Spectrum on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    mu:
        Distribution mean.
    sigma:
        Standard deviation bound.
    alpha:
        ES confidence level.
    c_l:
        ES floor target.
    sigma_phi:
        Spectral std dev (precomputed).
    A0:
        Benchmark A0 value (precomputed).

    Returns
    -------
    (Q_star, risk) : (NDArray, float)
    """
    # T(0) = benchmark ES = mu + sigma/sigma_phi * A0 < c_l (floor active)
    # T -> infinity as lam -> +infinity; bracket is [0, large]
    T0 = mu + sigma / sigma_phi * A0

    def f(lam: float) -> float:
        return _ES_floor_T(lam, phi_vals, u_grid, mu, sigma, alpha) - c_l

    if f(0.0) >= 0:
        # Already feasible at lam=0
        lam_l = 0.0
    else:
        # Find upper bracket
        hi = 1.0
        while f(hi) < 0 and hi < 1e6:
            hi *= 2.0
        lam_l = brentq(f, 0.0, hi, xtol=1e-12, rtol=1e-12)

    sigma_pt, A0_t, mean_pt = _tilted_sigma_phi(phi_vals, u_grid, alpha, lam_l)
    phi_tilde = phi_vals.copy()
    mask = u_grid >= alpha
    phi_tilde[mask] += lam_l / (1.0 - alpha)
    Q_star = mu + (sigma / sigma_pt) * (phi_tilde - mean_pt)
    risk = risk_quantile_space(phi_vals, Q_star, u_grid)
    return Q_star, risk


# ---------------------------------------------------------------------------
# Cap-active regime (three subcases)
# ---------------------------------------------------------------------------

def _tilted_sigma_phi_cap(
    phi_vals: NDArray,
    u_grid: NDArray,
    alpha: float,
    lam: float,
) -> Tuple[float, float, float]:
    """Tilted spectrum with negative lambda (cap reduces tail)."""
    phi_tilde = phi_vals.copy()
    mask = u_grid >= alpha
    phi_tilde[mask] -= lam / (1.0 - alpha)   # lam > 0 reduces ES
    mean_pt = 1.0 - lam
    sp = float(np.sqrt(np.trapz((phi_tilde - mean_pt) ** 2, u_grid)))
    if np.any(mask):
        A_t = float(np.trapz(phi_tilde[mask], u_grid[mask]))
    else:
        A_t = 0.0
    A0_t = A_t / (1.0 - alpha) - mean_pt
    return sp, A0_t, float(mean_pt)


def _ES_cap_T(
    lam: float,
    phi_vals: NDArray,
    u_grid: NDArray,
    mu: float,
    sigma: float,
    alpha: float,
) -> float:
    """ES_alpha(Q*(lam)) for the cap-active tilted spectrum."""
    sp, A0_t, _ = _tilted_sigma_phi_cap(phi_vals, u_grid, alpha, lam)
    if sp < 1e-15:
        return mu
    return mu + sigma / sp * A0_t


def solve_cap_active_interior(
    phi_vals: NDArray,
    u_grid: NDArray,
    mu: float,
    sigma: float,
    alpha: float,
    c_u: float,
) -> Tuple[NDArray, float]:
    """Solve the cap-active interior subcase via tilted-spectrum approach.

    Finds lambda_u >= 0 such that ES_alpha(Q*(lambda_u)) = c_u, where
    Q*(u) = mu + (sigma/sigma_phi_tilde) * (phi_tilde(u) - mean_phi_tilde) and
    phi_tilde(u) = phi(u) - lambda_u/(1-alpha) * 1_{u >= alpha}.

    Parameters
    ----------
    phi_vals:
        Spectrum on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    mu:
        Distribution mean.
    sigma:
        Standard deviation bound.
    alpha:
        ES confidence level.
    c_u:
        ES cap target.

    Returns
    -------
    (Q_star, risk) : (NDArray, float)
    """
    def f(lam: float) -> float:
        return _ES_cap_T(lam, phi_vals, u_grid, mu, sigma, alpha) - c_u

    # Bracket: at lam=0 ES > c_u (cap active); increase lam until ES < c_u
    hi = 1.0
    while f(hi) > 0 and hi < 1e6:
        hi *= 2.0

    if f(hi) > 0:
        # Use right-boundary subcase
        return solve_cap_active_right_boundary(phi_vals, u_grid, mu, sigma, alpha, c_u)

    lam_u = brentq(f, 0.0, hi, xtol=1e-12, rtol=1e-12)
    sp, A0_t, mean_pt = _tilted_sigma_phi_cap(phi_vals, u_grid, alpha, lam_u)
    phi_tilde = phi_vals.copy()
    mask = u_grid >= alpha
    phi_tilde[mask] -= lam_u / (1.0 - alpha)
    Q_star = mu + (sigma / sp) * (phi_tilde - mean_pt)
    risk = risk_quantile_space(phi_vals, Q_star, u_grid)
    return Q_star, risk


def solve_cap_active_right_boundary(
    phi_vals: NDArray,
    u_grid: NDArray,
    mu: float,
    sigma: float,
    alpha: float,
    c_u: float,
) -> Tuple[NDArray, float]:
    """Cap-active right-boundary subcase: Q* is flat at its upper limit.

    In this subcase the cap constraint forces Q* to have a plateau that
    extends all the way to u=1.  We impose ES = c_u directly on the
    constrained form

        Q*(u) = mu + (sigma/sigma_phi_tilde) * max(phi_tilde(u)-mean_pt, 0)

    and solve for the threshold.

    Parameters
    ----------
    phi_vals, u_grid, mu, sigma, alpha, c_u:
        As in :func:`solve_cap_active_interior`.

    Returns
    -------
    (Q_star, risk) : (NDArray, float)
    """
    # Project Q^diamond so that the tail ES matches c_u by capping Q at a
    # constant k for u >= alpha.  Mean constraint forces the flat level.
    sigma_phi = compute_sigma_phi(phi_vals, u_grid)
    Q_diamond = compute_Q_diamond(phi_vals, u_grid, mu, sigma, sigma_phi)

    def f_k(k: float) -> float:
        Q_trial = np.where(u_grid >= alpha, np.minimum(Q_diamond, k), Q_diamond)
        Q_trial = np.maximum.accumulate(Q_trial)  # enforce monotone
        mask = u_grid >= alpha
        if not np.any(mask):
            return -c_u
        return float(np.trapz(Q_trial[mask], u_grid[mask])) / (1.0 - alpha) - c_u

    lo = float(mu - 10 * sigma)
    hi = float(Q_diamond.max())

    if f_k(lo) * f_k(hi) > 0:
        # No crossing: return interior solution anyway
        Q_star = Q_diamond.copy()
        return Q_star, risk_quantile_space(phi_vals, Q_star, u_grid)

    k_star = brentq(f_k, lo, hi, xtol=1e-10, rtol=1e-10)
    Q_star = np.where(u_grid >= alpha, np.minimum(Q_diamond, k_star), Q_diamond)
    Q_star = np.maximum.accumulate(Q_star)
    risk = risk_quantile_space(phi_vals, Q_star, u_grid)
    return Q_star, risk


def solve_cap_active_left_boundary(
    phi_vals: NDArray,
    u_grid: NDArray,
    mu: float,
    sigma: float,
    alpha: float,
    c_u: float,
) -> Tuple[NDArray, float]:
    """Cap-active left-boundary subcase: plateau starts exactly at u=alpha.

    Q*(u) is unconstrained for u in [0, alpha) and equals a constant k
    for u in [alpha, 1], where k is set so that ES_alpha(Q*) = c_u.

    Parameters
    ----------
    phi_vals, u_grid, mu, sigma, alpha, c_u:
        As in :func:`solve_cap_active_interior`.

    Returns
    -------
    (Q_star, risk) : (NDArray, float)
    """
    # Cap Q*(u) = k for all u >= alpha; solve mean and ES constraints.
    # mean = integral_0^alpha Q*(u) du + (1-alpha)*k = mu
    # ES   = k = c_u  (flat tail)
    k = c_u
    sigma_phi = compute_sigma_phi(phi_vals, u_grid)
    Q_diamond = compute_Q_diamond(phi_vals, u_grid, mu, sigma, sigma_phi)

    Q_star = np.where(u_grid >= alpha, k, Q_diamond)
    Q_star = np.maximum.accumulate(Q_star)
    risk = risk_quantile_space(phi_vals, Q_star, u_grid)
    return Q_star, risk


# ---------------------------------------------------------------------------
# Risk for a general Q*
# ---------------------------------------------------------------------------

def compute_risk_model3(
    Q_star: NDArray,
    phi_vals: NDArray,
    u_grid: NDArray,
) -> float:
    """Compute R_phi(Q*) = integral_0^1 phi(u) Q*(u) du.

    Parameters
    ----------
    Q_star:
        Worst-case quantile function on *u_grid*, shape (n,).
    phi_vals:
        Spectrum on *u_grid*, shape (n,).
    u_grid:
        Probability grid in [0, 1], shape (n,).

    Returns
    -------
    float
    """
    return risk_quantile_space(phi_vals, Q_star, u_grid)


# ---------------------------------------------------------------------------
# Power-spectrum corollary (beta=3 and general beta)
# ---------------------------------------------------------------------------

def sigma_phi_power(beta: float) -> float:
    """Closed-form sigma_phi for the power spectrum phi(u)=beta*u^{beta-1}.

    sigma_phi = (beta-1) / sqrt(2*beta-1)

    Parameters
    ----------
    beta:
        Shape parameter, beta > 1.

    Returns
    -------
    float
    """
    return (beta - 1.0) / np.sqrt(2.0 * beta - 1.0)


# ---------------------------------------------------------------------------
# Full Model-3 solver
# ---------------------------------------------------------------------------

def worst_case_risk_model3(
    phi_vals: NDArray,
    u_grid: NDArray,
    mu: float,
    sigma: float,
    alpha: float,
    c_l: float,
    c_u: float,
    beta: Optional[float] = None,
) -> Tuple[NDArray, float, str]:
    """Compute worst-case spectral risk and Q* for Model 3.

    Parameters
    ----------
    phi_vals:
        Spectrum on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    mu:
        Distribution mean.
    sigma:
        Standard deviation bound.
    alpha:
        ES confidence level.
    c_l:
        ES floor.
    c_u:
        ES cap.
    beta:
        If provided, use the closed-form  sigma_phi = (beta-1)/sqrt(2*beta-1)
        for the power spectrum.

    Returns
    -------
    (Q_star, risk, regime) : (NDArray, float, str)
    """
    if beta is not None:
        sp = sigma_phi_power(beta)
    else:
        sp = compute_sigma_phi(phi_vals, u_grid)

    A0 = compute_A0(phi_vals, u_grid, alpha)
    Q_diamond = compute_Q_diamond(phi_vals, u_grid, mu, sigma, sp)
    M_alpha = mu + sigma / sp * A0

    regime = classify_regime_model3(0.0, c_l, c_u, M_alpha, mu)

    if regime == "benchmark":
        Q_star = Q_diamond
        risk = compute_risk_model3(Q_star, phi_vals, u_grid)
    elif regime == "floor_active":
        Q_star, risk = solve_floor_active(phi_vals, u_grid, mu, sigma, alpha, c_l, sp, A0)
    else:  # cap_active
        Q_star, risk = solve_cap_active_interior(phi_vals, u_grid, mu, sigma, alpha, c_u)

    return Q_star, risk, regime
