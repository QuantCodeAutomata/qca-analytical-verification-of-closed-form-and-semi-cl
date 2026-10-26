"""
algorithms.py
-------------
Projection-based proximal augmented-Lagrangian (ALM) algorithms for
computing worst-case spectral risk measures.

Two classes are provided:

* :class:`QuantileSpaceALM` – operates directly on the quantile function Q(u),
  enforcing monotonicity via the Pool-Adjacent-Violators (PAV) algorithm.

* :class:`DensitySpaceALM` – operates on the density f(x) >= 0, computes the
  CDF by the trapezoidal rule, and enforces non-negativity via clipping.

Both classes use an outer augmented-Lagrangian loop (dual-multiplier updates)
and an inner projected-gradient loop with Armijo backtracking.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Utility projections
# ---------------------------------------------------------------------------

def weighted_PAV(y: NDArray, weights: NDArray) -> NDArray:
    """Weighted Pool-Adjacent-Violators (PAV) algorithm for isotonic regression.

    Computes the non-decreasing sequence  Q*  that minimises
    sum_i  weights_i * (Q_i - y_i)^2.

    Parameters
    ----------
    y:
        Input vector to project onto the set of non-decreasing sequences.
    weights:
        Positive weight vector, same shape as *y*.

    Returns
    -------
    NDArray
        Isotonic (non-decreasing) projection of *y*, same shape.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(y)

    # Pool accumulation: each block stores (weighted_sum_y, sum_weights, size)
    sums_y = y * w
    sums_w = w.copy()
    blocks = list(range(n))  # block index of each element
    block_avg = (sums_y / sums_w).tolist()  # current block averages
    block_wy = sums_y.tolist()
    block_ww = sums_w.tolist()
    block_size = [1] * n

    # Merge adjacent violating blocks
    active = list(range(n))
    pool: list[dict] = [{"avg": y[i] * w[i] / w[i], "wy": y[i] * w[i], "ww": w[i], "idx": [i]}
                        for i in range(n)]

    i = 0
    while i < len(pool) - 1:
        if pool[i]["avg"] > pool[i + 1]["avg"]:
            # Merge i and i+1
            merged_wy = pool[i]["wy"] + pool[i + 1]["wy"]
            merged_ww = pool[i]["ww"] + pool[i + 1]["ww"]
            merged_avg = merged_wy / merged_ww
            merged_idx = pool[i]["idx"] + pool[i + 1]["idx"]
            pool[i] = {"avg": merged_avg, "wy": merged_wy, "ww": merged_ww, "idx": merged_idx}
            pool.pop(i + 1)
            # Check the block to the left again
            if i > 0:
                i -= 1
        else:
            i += 1

    result = np.empty(n)
    for block in pool:
        avg = block["avg"]
        for idx in block["idx"]:
            result[idx] = avg
    return result


def project_nonneg(x: NDArray) -> NDArray:
    """Project *x* onto the non-negative orthant: max(x, 0).

    Parameters
    ----------
    x:
        Input array.

    Returns
    -------
    NDArray
        Element-wise maximum with 0.
    """
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Quantile-Space ALM
# ---------------------------------------------------------------------------

class QuantileSpaceALM:
    """Augmented-Lagrangian method in quantile space.

    Solves::

        max_Q  h * sum_i phi(u_i) * Q_i
        s.t.   h * sum_i Q_i            = mu_target         (mean)
               h * sum_i Q_i^2          <= var_budget        (variance)
               Q_1 <= Q_2 <= ... <= Q_n                     (monotone)

    using a proximal augmented Lagrangian with PAV inner projection.

    Parameters
    ----------
    n_grid:
        Number of grid points.
    phi_func:
        Spectrum function phi(u), shape (n,) -> (n,).
    mu_target:
        Mean constraint (equality).
    var_budget:
        Variance budget sigma^2 (inequality).
    tau:
        Augmented Lagrangian penalty parameter rho.
    omega:
        Penalty growth factor (multiply rho by omega each outer iteration).
    inner_tol_func:
        Callable(k) -> float: inner tolerance at outer iteration k.
    outer_tol:
        Outer convergence tolerance.
    """

    def __init__(
        self,
        n_grid: int,
        phi_func: Callable[[NDArray], NDArray],
        mu_target: float,
        var_budget: float,
        tau: float = 1.0,
        omega: float = 1.2,
        inner_tol_func: Callable[[int], float] = lambda k: 1.0 / (k + 1) ** 2,
        outer_tol: float = 1e-6,
    ) -> None:
        self.n = n_grid
        self.h = 1.0 / n_grid
        self.u_grid = np.linspace(0.5 / n_grid, 1.0 - 0.5 / n_grid, n_grid)
        self.phi_vals = np.asarray(phi_func(self.u_grid), dtype=float)
        self.mu_target = mu_target
        self.var_budget = var_budget
        self.tau = tau
        self.omega = omega
        self.inner_tol_func = inner_tol_func
        self.outer_tol = outer_tol

    # ------------------------------------------------------------------
    def _objective(
        self,
        Q: NDArray,
        gamma: float,
        lambda_v: float,
        Q_prev: NDArray,
    ) -> float:
        """Augmented Lagrangian objective (to minimise; negate risk).

        L_aug(Q) = -h*phi^T*Q
                  + gamma*(h*sum(Q) - mu)
                  + (1/(2*rho)) * max(0, lambda_v + rho*(h*||Q||^2 - var_budget))^2
                  - lambda_v^2/(2*rho)
                  + (tau/2)*||Q - Q_prev||^2   (proximal term)
        """
        h = self.h
        rho = self.tau

        risk_term = -h * np.dot(self.phi_vals, Q)
        mean_res = h * np.sum(Q) - self.mu_target
        var_res = h * np.dot(Q, Q) - self.var_budget

        penalty_v = max(0.0, float(lambda_v) + rho * float(var_res))

        aug_lag = (risk_term
                   + float(gamma) * float(mean_res)
                   + (0.5 / rho) * penalty_v ** 2
                   - float(lambda_v) ** 2 / (2.0 * rho)
                   + 0.5 * self.tau * float(np.dot(Q - Q_prev, Q - Q_prev)))
        return float(aug_lag)

    # ------------------------------------------------------------------
    def _gradient(
        self,
        Q: NDArray,
        gamma: float,
        lambda_v: float,
        Q_prev: NDArray,
    ) -> NDArray:
        """Gradient of the augmented Lagrangian with respect to Q."""
        h = self.h
        rho = self.tau

        var_res = h * np.dot(Q, Q) - self.var_budget
        pen_v_scalar = max(0.0, float(lambda_v) + rho * float(var_res))

        grad = (-h * self.phi_vals
                + float(gamma) * h * np.ones(self.n)
                + pen_v_scalar * rho * 2.0 * h * Q
                + self.tau * (Q - Q_prev))
        return grad

    # ------------------------------------------------------------------
    def _inner_solve(
        self,
        Q_prev: NDArray,
        gamma: float,
        lambda_v: float,
        epsilon_k: float,
    ) -> NDArray:
        """Inner projected gradient with Armijo backtracking.

        Minimises the augmented Lagrangian (with proximal term) over the set
        of non-decreasing sequences, starting from *Q_prev*.
        """
        Q = Q_prev.copy()
        step = 1.0
        weights = np.ones(self.n) * self.h

        for _ in range(2000):
            g = self._gradient(Q, gamma, lambda_v, Q_prev)
            grad_norm = float(np.linalg.norm(g))
            if grad_norm < epsilon_k:
                break

            # Armijo line search
            f0 = self._objective(Q, gamma, lambda_v, Q_prev)
            for _ in range(50):
                Q_trial = weighted_PAV(Q - step * g, weights)
                f_trial = self._objective(Q_trial, gamma, lambda_v, Q_prev)
                if f_trial <= f0 - 1e-4 * step * float(np.dot(g, Q - Q_trial)):
                    break
                step *= 0.5

            Q = Q_trial
            step = min(step * 1.1, 10.0)

        return Q

    # ------------------------------------------------------------------
    def _update_multipliers(
        self,
        Q_new: NDArray,
        gamma: float,
        lambda_v: float,
    ) -> Tuple[float, float]:
        """Dual ascent: update gamma and lambda_v."""
        h = self.h
        rho = self.tau

        mean_res = h * np.sum(Q_new) - self.mu_target
        var_res = h * np.dot(Q_new, Q_new) - self.var_budget

        gamma_new = float(gamma) + rho * float(mean_res)
        lambda_v_new = max(0.0, float(lambda_v) + rho * float(var_res))
        return gamma_new, lambda_v_new

    # ------------------------------------------------------------------
    def _compute_residuals(
        self,
        Q: NDArray,
        gamma: float,
        lambda_v: float,
    ) -> Dict[str, float]:
        """Return stationarity, primal feasibility, and complementary-slackness residuals."""
        h = self.h
        rho = self.tau

        var_res = h * np.dot(Q, Q) - self.var_budget
        mean_res = h * np.sum(Q) - self.mu_target
        pen_v = max(0.0, float(lambda_v) + rho * float(var_res))

        grad = self._gradient(Q, float(gamma), float(lambda_v), Q)
        stat = float(np.linalg.norm(grad))

        return {
            "stationarity": stat,
            "mean_feasibility": abs(float(mean_res)),
            "var_feasibility": max(0.0, float(var_res)),
            "cs_var": abs(float(lambda_v) * float(var_res)),
        }

    # ------------------------------------------------------------------
    def solve(self, max_outer: int = 100) -> Tuple[NDArray, float, pd.DataFrame]:
        """Run the outer ALM loop.

        Parameters
        ----------
        max_outer:
            Maximum number of outer (dual) iterations.

        Returns
        -------
        (Q_star, risk, diagnostics_df) : (NDArray, float, pd.DataFrame)
        """
        Q = np.zeros(self.n)
        gamma = 0.0
        lambda_v = 0.0
        rows = []

        for k in range(max_outer):
            eps_k = self.inner_tol_func(k)
            Q = self._inner_solve(Q, gamma, lambda_v, eps_k)
            gamma, lambda_v = self._update_multipliers(Q, gamma, lambda_v)
            resids = self._compute_residuals(Q, gamma, lambda_v)
            risk = float(self.h * np.dot(self.phi_vals, Q))
            rows.append({"outer_iter": k, "risk": risk, **resids})
            self.tau *= self.omega

            converged = all(v < self.outer_tol for v in resids.values())
            if converged:
                break

        df = pd.DataFrame(rows)
        return Q, risk, df


# ---------------------------------------------------------------------------
# Density-Space ALM
# ---------------------------------------------------------------------------

class DensitySpaceALM:
    """Augmented-Lagrangian method in density/CDF space.

    Solves::

        max_f  integral phi(F(x)) * x * f(x) dx
        s.t.   integral f(x) dx         = 1           (normalisation)
               integral x * f(x) dx     = mean_target (mean)
               integral f(x)^2 dx       <= l2_budget  (L2 constraint)
               f(x) >= 0                              (non-negativity)

    where  F_i = h*(sum_{j<i} f_j + f_i/2)  (trapezoidal CDF).

    Parameters
    ----------
    n_grid:
        Number of grid points on the support.
    phi_func:
        Spectrum function phi(u), shape (n,) -> (n,).
    mean_target:
        Mean constraint.
    l2_budget:
        L2-norm budget for the density.
    tau:
        Augmented Lagrangian penalty.
    omega:
        Penalty growth factor.
    inner_tol_func:
        Inner tolerance schedule.
    outer_tol:
        Outer convergence tolerance.
    """

    def __init__(
        self,
        n_grid: int,
        phi_func: Callable[[NDArray], NDArray],
        mean_target: float,
        l2_budget: float,
        tau: float = 1.0,
        omega: float = 1.2,
        inner_tol_func: Callable[[int], float] = lambda k: 1.0 / (k + 1) ** 2,
        outer_tol: float = 1e-6,
    ) -> None:
        self.n = n_grid
        self.x_grid = np.linspace(0.0, 1.0, n_grid)
        self.h = self.x_grid[1] - self.x_grid[0]
        self.phi_func = phi_func
        self.mean_target = mean_target
        self.l2_budget = l2_budget
        self.tau = tau
        self.omega = omega
        self.inner_tol_func = inner_tol_func
        self.outer_tol = outer_tol

    # ------------------------------------------------------------------
    def _discrete_CDF(self, f: NDArray) -> NDArray:
        """Trapezoidal CDF: F_i = h*(sum_{j<i} f_j + f_i/2).

        Parameters
        ----------
        f:
            Density array, shape (n,).

        Returns
        -------
        NDArray
            CDF array in [0, 1], shape (n,).
        """
        h = self.h
        F = np.zeros(self.n)
        for i in range(1, self.n):
            F[i] = F[i - 1] + h * (f[i - 1] + f[i]) / 2.0
        F = np.clip(F, 0.0, 1.0)
        return F

    # ------------------------------------------------------------------
    def _compute_risk(self, f: NDArray) -> float:
        """Spectral risk: integral phi(F(x)) * x * f(x) dx."""
        F = self._discrete_CDF(f)
        phi_F = np.asarray(self.phi_func(F), dtype=float)
        return float(np.trapz(phi_F * self.x_grid * f, self.x_grid))

    # ------------------------------------------------------------------
    def _objective(
        self,
        f: NDArray,
        gamma_norm: float,
        gamma_mean: float,
        lambda_l2: float,
        f_prev: NDArray,
    ) -> float:
        """Augmented Lagrangian (minimisation form)."""
        h = self.h
        rho = self.tau

        risk = self._compute_risk(f)
        norm_res = h * np.sum(f) - 1.0
        mean_res = h * np.dot(self.x_grid, f) - self.mean_target
        l2_res = h * np.dot(f, f) - self.l2_budget
        pen_l2 = max(0.0, float(lambda_l2) + rho * float(l2_res))

        return (-float(risk)
                + float(gamma_norm) * float(norm_res)
                + float(gamma_mean) * float(mean_res)
                + (0.5 / rho) * pen_l2 ** 2
                - float(lambda_l2) ** 2 / (2.0 * rho)
                + 0.5 * self.tau * float(np.dot(f - f_prev, f - f_prev)))

    # ------------------------------------------------------------------
    def _gradient(
        self,
        f: NDArray,
        gamma_norm: float,
        gamma_mean: float,
        lambda_l2: float,
        f_prev: NDArray,
    ) -> NDArray:
        """Numerical gradient of the augmented Lagrangian w.r.t. f."""
        eps = 1e-5
        grad = np.zeros(self.n)
        f0 = self._objective(f, gamma_norm, gamma_mean, lambda_l2, f_prev)
        for i in range(self.n):
            f_pert = f.copy()
            f_pert[i] += eps
            grad[i] = (self._objective(f_pert, gamma_norm, gamma_mean, lambda_l2, f_prev) - f0) / eps
        return grad

    # ------------------------------------------------------------------
    def _inner_solve(
        self,
        f_prev: NDArray,
        gamma_norm: float,
        gamma_mean: float,
        lambda_l2: float,
        epsilon_k: float,
    ) -> NDArray:
        """Inner projected gradient descent with Armijo backtracking."""
        f = project_nonneg(f_prev.copy())
        step = 0.01

        for _ in range(500):
            g = self._gradient(f, gamma_norm, gamma_mean, lambda_l2, f_prev)
            if float(np.linalg.norm(g)) < epsilon_k:
                break
            f0 = self._objective(f, gamma_norm, gamma_mean, lambda_l2, f_prev)
            for _ in range(50):
                f_trial = project_nonneg(f - step * g)
                f_trial_val = self._objective(f_trial, gamma_norm, gamma_mean, lambda_l2, f_prev)
                if f_trial_val <= f0 - 1e-4 * step * float(np.dot(g, f - f_trial)):
                    break
                step *= 0.5
            f = f_trial
            step = min(step * 1.1, 1.0)

        return project_nonneg(f)

    # ------------------------------------------------------------------
    def _update_multipliers(
        self,
        f_new: NDArray,
        gamma_norm: float,
        gamma_mean: float,
        lambda_l2: float,
    ) -> Tuple[float, float, float]:
        """Dual ascent update."""
        h = self.h
        rho = self.tau

        norm_res = h * np.sum(f_new) - 1.0
        mean_res = h * np.dot(self.x_grid, f_new) - self.mean_target
        l2_res = h * np.dot(f_new, f_new) - self.l2_budget

        gn_new = float(gamma_norm) + rho * float(norm_res)
        gm_new = float(gamma_mean) + rho * float(mean_res)
        ll_new = max(0.0, float(lambda_l2) + rho * float(l2_res))
        return gn_new, gm_new, ll_new

    # ------------------------------------------------------------------
    def _compute_residuals(
        self,
        f: NDArray,
        gamma_norm: float,
        gamma_mean: float,
        lambda_l2: float,
    ) -> Dict[str, float]:
        """Residuals: feasibility of norm, mean, l2 constraints."""
        h = self.h
        norm_res = abs(h * np.sum(f) - 1.0)
        mean_res = abs(h * np.dot(self.x_grid, f) - self.mean_target)
        l2_res = max(0.0, h * np.dot(f, f) - self.l2_budget)
        cs_l2 = abs(float(lambda_l2) * (h * np.dot(f, f) - self.l2_budget))
        return {
            "norm_feasibility": float(norm_res),
            "mean_feasibility": float(mean_res),
            "l2_feasibility": float(l2_res),
            "cs_l2": float(cs_l2),
        }

    # ------------------------------------------------------------------
    def solve(self, max_outer: int = 100) -> Tuple[NDArray, float, pd.DataFrame]:
        """Run the outer ALM loop.

        Parameters
        ----------
        max_outer:
            Maximum number of outer iterations.

        Returns
        -------
        (f_star, risk, diagnostics_df) : (NDArray, float, pd.DataFrame)
        """
        f = np.ones(self.n) / (self.h * self.n)
        gamma_norm = 0.0
        gamma_mean = 0.0
        lambda_l2 = 0.0
        rows = []

        for k in range(max_outer):
            eps_k = self.inner_tol_func(k)
            f = self._inner_solve(f, gamma_norm, gamma_mean, lambda_l2, eps_k)
            gamma_norm, gamma_mean, lambda_l2 = self._update_multipliers(
                f, gamma_norm, gamma_mean, lambda_l2
            )
            resids = self._compute_residuals(f, gamma_norm, gamma_mean, lambda_l2)
            risk = self._compute_risk(f)
            rows.append({"outer_iter": k, "risk": risk, **resids})
            self.tau *= self.omega

            converged = all(v < self.outer_tol for v in resids.values())
            if converged:
                break

        df = pd.DataFrame(rows)
        return f, risk, df
