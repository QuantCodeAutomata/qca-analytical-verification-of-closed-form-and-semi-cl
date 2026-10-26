"""
model2_wasserstein_es.py
------------------------
Model 2: Worst-case spectral risk under a Wasserstein ambiguity set with an
Expected-Shortfall (ES) stress constraint.

The ambiguity set is

    M2(Q_G, epsilon, alpha, c) = {
        Q non-decreasing on [0,1] :
            W_2(Q, Q_G) <= epsilon,          (Wasserstein radius)
            ES_alpha(Q) >= c                  (ES stress floor)
    }

where  W_2(Q, Q_G)^2 = integral_0^1 (Q(u) - Q_G(u))^2 du  (quantile-space
2-Wasserstein distance).

The optimiser takes the form

    Q*(u) = Q_G(u)  +  [lambda_c / (1-alpha)] * 1_{u >= alpha}
                    +  [lambda_w / sqrt(B)]    * phi(u)

with three regimes determined by how the constraints interact.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.spectral_risk import risk_quantile_space


# ---------------------------------------------------------------------------
# Primitive quantities
# ---------------------------------------------------------------------------

def compute_cG(
    Q_G_vals: NDArray,
    u_grid: NDArray,
    alpha: float,
) -> float:
    """Expected Shortfall of the reference distribution Q_G at level alpha.

    c_G  =  1/(1-alpha) * integral_alpha^1 Q_G(u) du.

    Parameters
    ----------
    Q_G_vals:
        Reference quantile function evaluated on *u_grid*, shape (n,).
    u_grid:
        Probability grid in [0, 1], shape (n,).
    alpha:
        ES confidence level in (0, 1).

    Returns
    -------
    float
    """
    mask = u_grid >= alpha
    if not np.any(mask):
        return float(Q_G_vals[-1])
    return float(np.trapz(Q_G_vals[mask], u_grid[mask])) / (1.0 - alpha)


def compute_A_B(
    phi_vals: NDArray,
    u_grid: NDArray,
    alpha: float,
) -> Tuple[float, float]:
    """Compute the spectral integrals A and B.

    A  =  integral_alpha^1  phi(u) du
    B  =  integral_0^1      phi(u)^2 du

    Parameters
    ----------
    phi_vals:
        Spectrum on *u_grid*, shape (n,).
    u_grid:
        Probability grid in [0, 1].
    alpha:
        ES confidence level.

    Returns
    -------
    (A, B) : (float, float)
    """
    phi_vals = np.asarray(phi_vals, dtype=float)
    u_grid = np.asarray(u_grid, dtype=float)

    mask = u_grid >= alpha
    A = float(np.trapz(phi_vals[mask], u_grid[mask])) if np.any(mask) else 0.0
    B = float(np.trapz(phi_vals ** 2, u_grid))
    return A, B


def compute_c_dagger(
    c_G: float,
    epsilon: float,
    alpha: float,
    A: float,
    B: float,
) -> float:
    """Maximum achievable ES within the Wasserstein ball.

    c_dagger  =  c_G  +  epsilon * A / ((1-alpha) * sqrt(B)).

    This is the ES obtained when all the Wasserstein budget is directed along
    the ES-maximising direction (lambda_c=0, lambda_w=epsilon).

    Parameters
    ----------
    c_G:
        ES of the reference distribution.
    epsilon:
        Wasserstein radius.
    alpha:
        ES confidence level.
    A:
        integral_alpha^1 phi(u) du.
    B:
        integral_0^1 phi(u)^2 du.

    Returns
    -------
    float
    """
    return c_G + epsilon * A / ((1.0 - alpha) * np.sqrt(B))


def classify_regime(
    c: float,
    c_G: float,
    c_dagger: float,
    epsilon: float,
    alpha: float,
) -> str:
    """Classify the constraint regime for the given ES target *c*.

    Parameters
    ----------
    c:
        ES stress target.
    c_G:
        ES of the reference distribution.
    c_dagger:
        Maximum achievable ES within the Wasserstein ball.
    epsilon:
        Wasserstein radius.
    alpha:
        ES confidence level.

    Returns
    -------
    str
        One of ``'slack'``, ``'active'``, or ``'infeasible'``.
    """
    if c <= c_G:
        return "slack"
    elif c <= c_dagger:
        return "active"
    else:
        return "infeasible"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_model2(
    c: float,
    Q_G_vals: NDArray,
    phi_vals: NDArray,
    u_grid: NDArray,
    alpha: float,
    epsilon: float,
) -> Tuple[float, float, str]:
    """Compute Lagrange multipliers (lambda_c, lambda_w) and classify regime.

    Parameters
    ----------
    c:
        ES stress target (floor).
    Q_G_vals:
        Reference quantile function on *u_grid*.
    phi_vals:
        Spectrum on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    alpha:
        ES confidence level.
    epsilon:
        Wasserstein radius.

    Returns
    -------
    (lambda_c, lambda_w, regime) : (float, float, str)
        Lagrange multipliers and regime label.  *lambda_c* = 0 in the slack
        regime and both are NaN in the infeasible regime.
    """
    A, B = compute_A_B(phi_vals, u_grid, alpha)
    c_G = compute_cG(Q_G_vals, u_grid, alpha)
    c_dagger = compute_c_dagger(c_G, epsilon, alpha, A, B)
    regime = classify_regime(c, c_G, c_dagger, epsilon, alpha)

    if regime == "slack":
        return 0.0, float(epsilon), regime

    if regime == "infeasible":
        return float("nan"), float("nan"), regime

    # Active regime
    delta = (c - c_G) * (1.0 - alpha)            # = lambda_c + lambda_w*A/sqrt(B)
    denom = B - A ** 2 / (1.0 - alpha)
    z_sq = (epsilon ** 2 - delta ** 2 / (1.0 - alpha)) / denom
    z = np.sqrt(max(z_sq, 0.0))                   # lambda_w / sqrt(B)
    lambda_w = z * np.sqrt(B)
    lambda_c = delta - z * A

    return float(lambda_c), float(lambda_w), regime


# ---------------------------------------------------------------------------
# Worst-case quantile function
# ---------------------------------------------------------------------------

def construct_Q_star_model2(
    Q_G_vals: NDArray,
    phi_vals: NDArray,
    u_grid: NDArray,
    alpha: float,
    lambda_c: float,
    lambda_w: float,
) -> NDArray:
    """Build Q*(u) on the given grid.

    Q*(u) = Q_G(u)
            + lambda_c / (1-alpha) * 1_{u >= alpha}
            + lambda_w / sqrt(B) * phi(u)

    Parameters
    ----------
    Q_G_vals:
        Reference quantile function on *u_grid*.
    phi_vals:
        Spectrum on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    alpha:
        ES confidence level.
    lambda_c:
        ES-stress multiplier.
    lambda_w:
        Wasserstein multiplier.

    Returns
    -------
    NDArray
        Q*(u) on *u_grid*, shape (n,).
    """
    phi_vals = np.asarray(phi_vals, dtype=float)
    Q_G_vals = np.asarray(Q_G_vals, dtype=float)
    u_grid = np.asarray(u_grid, dtype=float)

    B = float(np.trapz(phi_vals ** 2, u_grid))
    sqrt_B = np.sqrt(max(B, 1e-30))

    es_indicator = (u_grid >= alpha).astype(float)
    return Q_G_vals + lambda_c / (1.0 - alpha) * es_indicator + lambda_w / sqrt_B * phi_vals


# ---------------------------------------------------------------------------
# Risk value
# ---------------------------------------------------------------------------

def compute_risk_model2(
    phi_vals: NDArray,
    Q_G_vals: NDArray,
    u_grid: NDArray,
    lambda_c: float,
    lambda_w: float,
    alpha: float,
    A: float,
    B: float,
) -> float:
    """Compute the worst-case spectral risk R_phi(Q*).

    R_phi(Q*)  =  R_phi(Q_G)
                 + lambda_c * A / (1-alpha)
                 + lambda_w * sqrt(B)

    Parameters
    ----------
    phi_vals:
        Spectrum on *u_grid*.
    Q_G_vals:
        Reference quantile function on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    lambda_c:
        ES multiplier from :func:`calibrate_model2`.
    lambda_w:
        Wasserstein multiplier from :func:`calibrate_model2`.
    alpha:
        ES confidence level.
    A:
        integral_alpha^1 phi(u) du.
    B:
        integral_0^1 phi(u)^2 du.

    Returns
    -------
    float
        Worst-case risk value.
    """
    R_G = risk_quantile_space(phi_vals, Q_G_vals, u_grid)
    return R_G + lambda_c * A / (1.0 - alpha) + lambda_w * np.sqrt(B)


# ---------------------------------------------------------------------------
# Numerical cross-check via cvxpy
# ---------------------------------------------------------------------------

def cvxpy_crosscheck_model2(
    phi_vals: NDArray,
    Q_G_vals: NDArray,
    u_grid: NDArray,
    alpha: float,
    epsilon: float,
    c: float,
) -> dict:
    """Validate :func:`compute_risk_model2` against a cvxpy SOCP.

    Parameters
    ----------
    phi_vals:
        Spectrum on *u_grid*.
    Q_G_vals:
        Reference quantile on *u_grid*.
    u_grid:
        Probability grid in [0, 1].
    alpha:
        ES confidence level.
    epsilon:
        Wasserstein radius.
    c:
        ES stress target.

    Returns
    -------
    dict with keys 'analytic', 'cvxpy', 'abs_diff', 'regime'.
    """
    import cvxpy as cp

    phi_vals = np.asarray(phi_vals, dtype=float)
    Q_G_vals = np.asarray(Q_G_vals, dtype=float)
    u_grid = np.asarray(u_grid, dtype=float)
    n = len(u_grid)
    h = float(u_grid[1] - u_grid[0]) if n > 1 else 1.0

    A, B = compute_A_B(phi_vals, u_grid, alpha)
    c_G = compute_cG(Q_G_vals, u_grid, alpha)
    c_dagger = compute_c_dagger(c_G, epsilon, alpha, A, B)
    regime = classify_regime(c, c_G, c_dagger, epsilon, alpha)

    lambda_c, lambda_w, _ = calibrate_model2(c, Q_G_vals, phi_vals, u_grid, alpha, epsilon)

    if regime == "infeasible":
        return {"analytic": float("nan"), "cvxpy": float("nan"), "abs_diff": 0.0, "regime": regime}

    analytic = compute_risk_model2(phi_vals, Q_G_vals, u_grid, lambda_c, lambda_w, alpha, A, B)

    Q = cp.Variable(n)
    delta_Q = Q - Q_G_vals
    es_mask = u_grid >= alpha
    objective = cp.Maximize(h * (phi_vals @ Q))
    constraints = [
        cp.sum_squares(delta_Q) * h <= epsilon ** 2,
        Q[:-1] <= Q[1:],
    ]
    if np.any(es_mask):
        idx = np.where(es_mask)[0]
        constraints.append(h * cp.sum(Q[idx]) / (1.0 - alpha) >= c)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    cvxpy_val = float(prob.value) if prob.value is not None else float("nan")

    return {
        "analytic": analytic,
        "cvxpy": cvxpy_val,
        "abs_diff": abs(analytic - cvxpy_val),
        "regime": regime,
    }
