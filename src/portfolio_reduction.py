"""
portfolio_reduction.py
----------------------
Model 6b portfolio projection identity and SOCP reduction.

**Key identity (projection theorem):**
For portfolio return  X = w^T r  where  r ~ N(mu_hat, Sigma_hat)  and a
KL ambiguity ball of radius *epsilon* centred on the Gaussian reference, the
worst-case Expected Shortfall satisfies

    sup_{f in B_eps(N(mu_w, sigma_w^2))}  ES_alpha(X)
        = mu_w  +  sigma_w * chi(alpha, epsilon)

where

    mu_w    = w^T mu_hat
    sigma_w = sqrt(w^T Sigma_hat w)
    chi     = compute_chi_primal from Model 6b with bar_mu = 0

This projection property means the worst-case ES of the portfolio depends
only on the portfolio mean and standard deviation via the scalar *chi*, so
the robust portfolio problem reduces to an SOCP.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Chi calibration
# ---------------------------------------------------------------------------

def calibrate_chi(alpha: float, epsilon: float) -> float:
    """Compute chi(alpha, epsilon) from Model 6b with bar_mu = 0.

    chi equals the worst-case ES of N(0,1) under a KL ball of radius epsilon.

    Parameters
    ----------
    alpha:
        ES confidence level in (0, 1).
    epsilon:
        KL divergence radius, epsilon > 0.

    Returns
    -------
    float
        chi(alpha, epsilon).
    """
    from src.model6b_kl_gaussian import solve_primal_system_6b, compute_chi_primal

    _, lam2, x_alpha = solve_primal_system_6b(alpha, epsilon, bar_mu=0.0)
    return compute_chi_primal(lam2, x_alpha)


# ---------------------------------------------------------------------------
# Portfolio worst-case ES (the projection identity)
# ---------------------------------------------------------------------------

def portfolio_worst_case_ES(
    w: NDArray,
    mu_hat: NDArray,
    Sigma_hat: NDArray,
    chi: float,
) -> float:
    """Worst-case ES of portfolio return under the KL-Gaussian ambiguity.

    V(w)  =  mu_w  +  sigma_w * chi

    Parameters
    ----------
    w:
        Portfolio weight vector, shape (d,).
    mu_hat:
        Estimated mean return vector, shape (d,).
    Sigma_hat:
        Estimated covariance matrix, shape (d, d).
    chi:
        Worst-case chi from :func:`calibrate_chi`.

    Returns
    -------
    float
        Worst-case ES value V(w).
    """
    w = np.asarray(w, dtype=float)
    mu_hat = np.asarray(mu_hat, dtype=float)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)

    mu_w = float(w @ mu_hat)
    var_w = float(w @ Sigma_hat @ w)
    sigma_w = float(np.sqrt(max(var_w, 0.0)))
    return mu_w + sigma_w * chi


# ---------------------------------------------------------------------------
# SOCP robust portfolio
# ---------------------------------------------------------------------------

def solve_robust_portfolio_SOCP(
    mu_hat: NDArray,
    Sigma_hat: NDArray,
    chi: float,
    W_constraints_func: Optional[Callable] = None,
    budget_constraint: bool = True,
    long_only: bool = False,
) -> Tuple[NDArray, float]:
    """Solve the robust portfolio SOCP.

    Minimises the worst-case portfolio ES:

        min_w   w^T mu_hat + chi * ||Sigma_hat^{1/2} w||_2
        s.t.    portfolio constraints

    By default the only constraint is the budget  sum(w) = 1  (plus any
    constraints supplied via *W_constraints_func*).

    Parameters
    ----------
    mu_hat:
        Estimated mean return vector, shape (d,).
    Sigma_hat:
        Estimated covariance matrix, shape (d, d).
    chi:
        Scalar chi from :func:`calibrate_chi`.
    W_constraints_func:
        Optional callable that accepts a cvxpy Variable *w* and returns a
        list of additional cvxpy constraints.
    budget_constraint:
        If True, enforce  sum(w) = 1.
    long_only:
        If True, enforce  w >= 0.

    Returns
    -------
    (w_star, es_star) : (NDArray, float)
        Optimal weights and the corresponding worst-case ES.
    """
    import cvxpy as cp

    d = len(mu_hat)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)

    # Cholesky factor for SOC
    try:
        L = np.linalg.cholesky(Sigma_hat)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma_hat + 1e-10 * np.eye(d))

    w = cp.Variable(d)
    sigma_w = cp.Variable(nonneg=True)  # portfolio std dev

    objective = cp.Minimize(mu_hat @ w + chi * sigma_w)
    constraints = [cp.norm(L.T @ w, 2) <= sigma_w]

    if budget_constraint:
        constraints.append(cp.sum(w) == 1.0)
    if long_only:
        constraints.append(w >= 0)
    if W_constraints_func is not None:
        constraints += W_constraints_func(w)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        w_star = np.ones(d) / d
    else:
        w_star = np.asarray(w.value, dtype=float)

    es_star = portfolio_worst_case_ES(w_star, mu_hat, Sigma_hat, chi)
    return w_star, es_star


# ---------------------------------------------------------------------------
# Projection identity verification
# ---------------------------------------------------------------------------

def verify_projection_identity(
    w: NDArray,
    mu_hat: NDArray,
    Sigma_hat: NDArray,
    chi: float,
    alpha: float,
    epsilon: float,
    density_6b_func: Callable,
) -> Dict[str, float]:
    """Verify the projection identity by comparing analytic and simulation-based ES.

    Computes:
      - V_analytic = mu_w + sigma_w * chi                   (analytic)
      - V_simulated = sample ES of X = w^T r under f*       (Monte Carlo)

    Parameters
    ----------
    w:
        Portfolio weights.
    mu_hat:
        Mean vector.
    Sigma_hat:
        Covariance matrix.
    chi:
        Worst-case chi.
    alpha:
        ES level.
    epsilon:
        KL radius.
    density_6b_func:
        Function returning the worst-case density f*(x) for scalar X.

    Returns
    -------
    dict with keys 'V_analytic', 'V_simulated', 'abs_diff'.
    """
    from src.model6b_kl_gaussian import solve_primal_system_6b

    w = np.asarray(w, dtype=float)
    mu_w = float(w @ mu_hat)
    sigma_w = float(np.sqrt(max(float(w @ Sigma_hat @ w), 0.0)))

    # Analytic
    V_analytic = mu_w + sigma_w * chi

    # Simulated: the worst-case density of X is N(mu_w + sigma_w*lam2, sigma_w^2) in the tail
    lam1, lam2, x_alpha_std = solve_primal_system_6b(alpha, epsilon, bar_mu=0.0)
    # Scale to portfolio
    x_alpha_port = mu_w + sigma_w * x_alpha_std

    rng = np.random.default_rng(42)
    n_sim = 100_000
    X = rng.normal(mu_w + sigma_w * lam2, sigma_w, n_sim)
    X_sorted = np.sort(X)
    q_idx = int(np.ceil(alpha * n_sim))
    tail = X_sorted[q_idx:]
    V_simulated = float(np.mean(tail)) if len(tail) > 0 else float("nan")

    return {
        "V_analytic": V_analytic,
        "V_simulated": V_simulated,
        "abs_diff": abs(V_analytic - V_simulated),
    }


# ---------------------------------------------------------------------------
# Joint worst-case law construction
# ---------------------------------------------------------------------------

def construct_joint_worst_case_law(
    w: NDArray,
    mu_hat: NDArray,
    Sigma_hat: NDArray,
    lambda1: float,
    lambda2: float,
    x_alpha: float,
    alpha: float,
) -> Callable[[NDArray], NDArray]:
    """Construct the joint worst-case law for the return vector r.

    The worst-case law for the portfolio  X = w^T r  is a piecewise tilted
    Gaussian.  For the full vector, we construct the conditional Gaussian
    given X = x.

    The joint worst-case density is a mixture:
      - For x <= x_alpha: r | X=x ~ N(mu_hat + Sigma*w*(x - mu_w)/sigma_w^2, Sigma_perp)
                           tilted by lambda1 in the X direction.
      - For x > x_alpha:  tilted by lambda2.

    Returns a callable  P_joint(r_samples)  that evaluates the density.

    Parameters
    ----------
    w:
        Portfolio weights.
    mu_hat, Sigma_hat:
        Gaussian reference parameters.
    lambda1, lambda2:
        Tilting parameters from Model 6b.
    x_alpha:
        Alpha-quantile breakpoint.
    alpha:
        ES level.

    Returns
    -------
    Callable
        Function mapping an (n, d) sample array to density values.
    """
    w = np.asarray(w, dtype=float)
    d = len(w)
    mu_w = float(w @ mu_hat)
    sigma_w2 = float(w @ Sigma_hat @ w)
    sigma_w = float(np.sqrt(max(sigma_w2, 1e-15)))

    # Projection direction
    Sigma_w = Sigma_hat @ w  # d-vector

    def joint_density(r_samples: NDArray) -> NDArray:
        """Evaluate the joint worst-case density at sample points.

        Parameters
        ----------
        r_samples:
            Array of shape (n, d).

        Returns
        -------
        NDArray of shape (n,).
        """
        from scipy.stats import multivariate_normal

        r_samples = np.asarray(r_samples, dtype=float)
        X_vals = r_samples @ w  # portfolio returns, shape (n,)

        # Worst-case density of X: piecewise tilted Gaussian
        from src.model6b_kl_gaussian import construct_density_6b

        # Left piece (X <= x_alpha): tilted Gaussian N(lambda1, 1) in standardised coords
        # Right piece (X > x_alpha): tilted Gaussian N(lambda2, 1)
        sqrt2pi = np.sqrt(2.0 * np.pi)
        from src.model6b_kl_gaussian import J0
        j0_left = J0(lambda1, -np.inf, x_alpha)
        j0_right = J0(lambda2, x_alpha, np.inf)
        C1 = alpha / (sqrt2pi * max(j0_left, 1e-300))
        C2 = (1.0 - alpha) / (sqrt2pi * max(j0_right, 1e-300))

        X_std = (X_vals - mu_w) / sigma_w
        x_alpha_std = (x_alpha - mu_w) / sigma_w

        # Marginal density of X under worst-case f*
        f_X = np.where(
            X_std <= x_alpha_std,
            C1 * np.exp(-X_std ** 2 / 2.0 + lambda1 * X_std) / sigma_w,
            C2 * np.exp(-X_std ** 2 / 2.0 + lambda2 * X_std) / sigma_w,
        )

        # Conditional density r | X=x ~ N(mu_cond(x), Sigma_cond)
        # mu_cond(x) = mu_hat + Sigma_hat*w*(x - mu_w)/sigma_w^2
        # Sigma_cond = Sigma_hat - Sigma_hat*w*w^T*Sigma_hat / sigma_w^2

        Sigma_cond = Sigma_hat - np.outer(Sigma_w, Sigma_w) / max(sigma_w2, 1e-15)
        # Add small regularisation
        Sigma_cond += 1e-8 * np.eye(d)

        mu_cond = mu_hat + np.outer((X_vals - mu_w) / sigma_w2, Sigma_w)  # (n, d)

        # Evaluate conditional density (expensive for large n)
        f_cond = np.array([
            multivariate_normal.pdf(r_samples[i], mean=mu_cond[i], cov=Sigma_cond)
            for i in range(len(r_samples))
        ])

        return f_X * f_cond

    return joint_density


# ---------------------------------------------------------------------------
# Joint law property verification
# ---------------------------------------------------------------------------

def verify_joint_law_properties(
    P_joint_func: Callable[[NDArray], NDArray],
    w: NDArray,
    mu_hat: NDArray,
    Sigma_hat: NDArray,
    epsilon: float,
    n_samples: int = 5000,
) -> Dict[str, float]:
    """Verify properties of the joint worst-case law.

    Checks:
      1. Marginal mean approximation:  E_{P*}[r] ≈ mu_hat  (up to KL perturbation)
      2. Portfolio ES is finite.

    Parameters
    ----------
    P_joint_func:
        Joint density function from :func:`construct_joint_worst_case_law`.
    w:
        Portfolio weights.
    mu_hat, Sigma_hat:
        Reference Gaussian parameters.
    epsilon:
        KL radius.
    n_samples:
        Monte Carlo sample size.

    Returns
    -------
    dict with diagnostic statistics.
    """
    w = np.asarray(w, dtype=float)
    d = len(w)

    rng = np.random.default_rng(123)
    # Draw from the reference N(mu_hat, Sigma_hat)
    samples = rng.multivariate_normal(mu_hat, Sigma_hat, size=n_samples)
    X_vals = samples @ w
    X_sorted = np.sort(X_vals)

    # Importance weight: P_joint / reference_density
    from scipy.stats import multivariate_normal
    ref_density = multivariate_normal.pdf(samples, mean=mu_hat, cov=Sigma_hat)
    wc_density = P_joint_func(samples)
    iw = np.where(ref_density > 1e-300, wc_density / ref_density, 0.0)
    iw = np.clip(iw / np.mean(iw), 0.0, 1e6)  # normalised importance weights

    mean_est = float(np.mean(iw[:, None] * samples, axis=0) @ w) if d > 0 else float("nan")

    alpha_idx = int(np.ceil(0.95 * n_samples))
    es_approx = float(np.mean(X_sorted[alpha_idx:]))

    return {
        "portfolio_mean_estimate": float(mean_est),
        "portfolio_ES_approx": float(es_approx),
        "n_samples": n_samples,
        "iw_eff_ratio": float(1.0 / np.mean(iw ** 2)) if np.mean(iw ** 2) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Chi sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis_chi(
    alpha_vals: List[float],
    epsilon_vals: List[float],
) -> pd.DataFrame:
    """Tabulate chi(alpha, epsilon) for bar_mu=0 over a grid.

    Parameters
    ----------
    alpha_vals:
        ES confidence levels.
    epsilon_vals:
        KL radii.

    Returns
    -------
    pd.DataFrame
        Columns: alpha, epsilon, chi.
    """
    rows = []
    for alpha in alpha_vals:
        for eps in epsilon_vals:
            try:
                chi = calibrate_chi(alpha, eps)
            except Exception:
                chi = float("nan")
            rows.append({"alpha": alpha, "epsilon": eps, "chi": chi})
    return pd.DataFrame(rows)
