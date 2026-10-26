"""
exp3_algorithms.py
Projection-based proximal augmented-Lagrangian algorithm experiments
reproducing Table 2 of the paper.

Paper settings (Appendix E.4):
  phi(u) = 2u, tau = 1, omega = 0.5
  inner tol: epsilon_k = 2e-3 * (k+1)^{-1.6}
  outer stopping: feasibility, CS, and unit-step projected-gradient all < 3e-7
"""

import sys
sys.path.insert(0, '/workspace/project')

import json, csv, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.makedirs('/workspace/project/results', exist_ok=True)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


# ── Weighted Pool-Adjacent-Violators (isotonic regression, nondecreasing) ────

def weighted_PAV(y: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Isotonic regression (nondecreasing) via Pool-Adjacent-Violators.
    Returns the nondecreasing projection of y under weighted L2 norm.
    """
    n = len(y)
    if weights is None:
        weights = np.ones(n)
    y = y.astype(float).copy()
    w = weights.astype(float).copy()

    blocks = []
    for i in range(n):
        blk = [y[i], w[i], i, i]
        while blocks and blk[0] < blocks[-1][0]:
            prev = blocks.pop()
            total_w = prev[1] + blk[1]
            merged_mean = (prev[0] * prev[1] + blk[0] * blk[1]) / total_w
            blk = [merged_mean, total_w, prev[2], blk[3]]
        blocks.append(blk)

    result = np.empty(n)
    for blk in blocks:
        result[blk[2]:blk[3] + 1] = blk[0]
    return result


# ── Quantile-Space ALM ────────────────────────────────────────────────────────

def quantile_space_ALM(n: int = 80, mean_target: float = 0.0, var_budget: float = 1.0,
                       tau: float = 1.0, omega: float = 0.5,
                       outer_tol: float = 3e-7, max_outer: int = 1000) -> dict:
    """
    Proximal augmented-Lagrangian in quantile space.

    Maximize R_phi(Q) = h*sum(phi_i * Q_i) subject to:
      h*sum(Q_i) = mean_target        [equality,  multiplier gamma]
      h*sum(Q_i^2) <= var_budget      [inequality, multiplier lam >= 0]
      Q nondecreasing                 [structural, enforced by PAV projection]

    phi(u) = 2u.  Paper: n=80, mean=0, var=1, tau=1, omega=0.5.
    Target Table 2: ~60 iters, risk=0.57730534, stat~2.94e-7.
    """
    h = 1.0 / n
    u = (np.arange(n) + 0.5) / n
    phi = 2.0 * u

    Q = np.zeros(n)    # Q^0 = 0
    gamma = 0.0        # equality multiplier
    lam   = 0.0        # inequality multiplier (>= 0)

    risk_trace, stat_trace = [], []
    outer_iter = 0

    def inner_tol(k: int) -> float:
        return 2e-3 * (k + 1) ** (-1.6)

    # Augmented Lagrangian value (including proximal)
    def J_k(Q_v: np.ndarray, Q_prev: np.ndarray,
            gamma_v: float, lam_v: float) -> float:
        eq  = h * Q_v.sum() - mean_target
        ineq = h * Q_v.dot(Q_v) - var_budget
        aug  = max(0.0, lam_v + tau * ineq)
        return (- h * phi.dot(Q_v)
                + gamma_v * eq + (tau / 2.0) * eq**2
                + (1.0 / (2.0 * tau)) * (aug**2 - lam_v**2)
                + (omega / 2.0) * np.dot(Q_v - Q_prev, Q_v - Q_prev))

    # Gradient of J_k (includes proximal term omega*(Q - Q_prev))
    def grad_J_k(Q_v: np.ndarray, Q_prev: np.ndarray,
                 gamma_v: float, lam_v: float) -> np.ndarray:
        eq   = h * Q_v.sum() - mean_target
        ineq = h * Q_v.dot(Q_v) - var_budget
        aug  = max(0.0, lam_v + tau * ineq)
        g    = (-h * phi
                + (gamma_v + tau * eq) * h
                + aug * 2.0 * h * Q_v
                + omega * (Q_v - Q_prev))
        return g

    # Gradient WITHOUT proximal (used for outer stationarity check)
    def grad_outer(Q_v: np.ndarray, gamma_v: float, lam_v: float) -> np.ndarray:
        eq   = h * Q_v.sum() - mean_target
        ineq = h * Q_v.dot(Q_v) - var_budget
        aug  = max(0.0, lam_v + tau * ineq)
        return (-h * phi
                + (gamma_v + tau * eq) * h
                + aug * 2.0 * h * Q_v)

    for k in range(max_outer):
        Q_prev = Q.copy()
        tol_k  = inner_tol(k)

        # ── Inner loop: projected gradient with Armijo backtracking ──────────
        z = Q_prev.copy()
        for _inner in range(10000):
            grad_z = grad_J_k(z, Q_prev, gamma, lam)
            J_z    = J_k(z, Q_prev, gamma, lam)

            # Armijo backtracking: find delta such that
            #   J_k(v) <= J_k(z) + <grad_z, v-z> + ||v-z||^2 / (2*delta)
            delta = 1.0
            for _ in range(60):
                v   = weighted_PAV(z - delta * grad_z)
                J_v = J_k(v, Q_prev, gamma, lam)
                dv  = v - z
                if J_v <= J_z + grad_z.dot(dv) + dv.dot(dv) / (2.0 * delta):
                    break
                delta *= 0.5

            # Inner residual: e = grad_J_k(v) - grad_z + (z - v)/delta
            grad_v = grad_J_k(v, Q_prev, gamma, lam)
            e      = grad_v - grad_z + (z - v) / delta
            e_norm = np.linalg.norm(e)

            z = v
            if e_norm <= tol_k:
                break

        Q = z

        # ── Outer residuals ───────────────────────────────────────────────────
        eq_val   = h * Q.sum() - mean_target
        ineq_val = h * Q.dot(Q) - var_budget
        feas_res = abs(eq_val) + max(0.0, ineq_val)
        cs_res   = abs(lam * ineq_val)   # complementary slackness

        # Unit-step projected-gradient stationarity (no proximal)
        g_out    = grad_outer(Q, gamma, lam)
        stat_res = np.linalg.norm(Q - weighted_PAV(Q - g_out))

        risk_val = float(h * phi.dot(Q))
        risk_trace.append(risk_val)
        stat_trace.append(float(stat_res))
        outer_iter = k + 1

        if outer_iter % 50 == 0:
            print(f"  Q-ALM k={outer_iter:4d}: risk={risk_val:.8f} "
                  f"feas={feas_res:.2e} cs={cs_res:.2e} stat={stat_res:.2e}")

        # ── Multiplier updates ────────────────────────────────────────────────
        gamma += tau * eq_val
        lam    = max(0.0, lam + tau * ineq_val)

        if feas_res < outer_tol and cs_res < outer_tol and stat_res < outer_tol:
            break

    risk_final = float(h * phi.dot(Q))
    print(f"  Q-ALM converged: k={outer_iter}, risk={risk_final:.8f}, stat={stat_trace[-1]:.2e}")
    return {
        'outer_iters': outer_iter,
        'risk':        risk_final,
        'stat':        stat_trace[-1] if stat_trace else np.nan,
        'feas_res':    float(abs(h * Q.sum() - mean_target) + max(0.0, h * Q.dot(Q) - var_budget)),
        'cs_res':      float(abs(lam * (h * Q.dot(Q) - var_budget))),
        'risk_trace':  risk_trace,
        'stat_trace':  stat_trace,
        'Q_opt':       Q.tolist(),
    }


# ── Density-Space ALM ─────────────────────────────────────────────────────────

def density_space_ALM(n: int = 80, mean_target: float = 0.5, l2_budget: float = 2.0,
                      tau: float = 1.0, omega: float = 0.5,
                      outer_tol: float = 3e-7, max_outer: int = 3000) -> dict:
    """
    Proximal augmented-Lagrangian in density/CDF space.

    Maximize R_phi(F) = 1 - h*sum(F_i^2) subject to:
      h*sum(f_i) = 1                   [normalization, equality, mult g1]
      h*sum(x_i * f_i) = mean_target   [mean,          equality, mult g2]
      h*sum(f_i^2) <= l2_budget        [L2 density,  inequality, mult lam >= 0]
      f_i >= 0                          [nonnegativity, enforced by projection]

    CDF midpoint formula: F_i = h*(sum_{j<i} f_j + f_i/2).
    phi(u)=2u => Phi(t)=t^2 => R_phi(F) = b - integral Phi(F) dx
                                          = 1 - h*sum(F_i^2).

    Paper: n=80 and n=160, mean=1/2, L2=2, tau=1, omega=0.5.
    Table 2 targets: ~261 iters, risk~0.71870715 (n=80), ~0.71870693 (n=160).

    Implementation note: this ALM uses scipy L-BFGS-B for the inner
    sub-problem (bounds-constrained minimisation of J_k over f >= 0),
    which gives reliable convergence. The paper's 261-iteration count
    reflects a more efficient projected-gradient inner loop; our convergence
    is structurally correct but requires more outer steps to activate the
    L2 inequality (which starts inactive at f^0 = 1 with h*sum(f^2)=1 < 2).
    """
    from scipy.optimize import minimize as sp_minimize

    h = 1.0 / n
    x = (np.arange(n) + 0.5) / n     # midpoints in [0,1]

    def compute_F(f_v: np.ndarray) -> np.ndarray:
        """CDF midpoints: F_i = h*(sum_{j<i} f_j + f_i/2)."""
        cs = np.concatenate(([0.0], np.cumsum(f_v)[:-1]))
        return h * (cs + 0.5 * f_v)

    def compute_risk(f_v: np.ndarray) -> float:
        """R_phi(F) = 1 - h*sum(F_i^2)  for phi(u)=2u."""
        F = compute_F(f_v)
        return float(1.0 - h * F.dot(F))

    def _obj_grad_base(f_v: np.ndarray) -> np.ndarray:
        """Gradient of h*sum(F_i^2) w.r.t. f_i = h^2*(2*S_i - F_i)."""
        F = compute_F(f_v)
        S = np.cumsum(F[::-1])[::-1]     # S[i] = sum_{j>=i} F[j]
        return h * h * (2.0 * S - F)

    def J_k(f_v: np.ndarray, f_prev: np.ndarray,
            g1: float, g2: float, lam_v: float) -> float:
        """Augmented Lagrangian value (with proximal term)."""
        F      = compute_F(f_v)
        risk   = 1.0 - h * F.dot(F)
        eq1_r  = h * f_v.sum() - 1.0
        eq2_r  = h * x.dot(f_v) - mean_target
        ineq_r = h * f_v.dot(f_v) - l2_budget
        aug    = max(0.0, lam_v + tau * ineq_r)
        return (-risk
                + g1 * eq1_r + (tau / 2.0) * eq1_r**2
                + g2 * eq2_r + (tau / 2.0) * eq2_r**2
                + (1.0 / (2.0 * tau)) * (aug**2 - lam_v**2)
                + (omega / 2.0) * np.dot(f_v - f_prev, f_v - f_prev))

    def grad_J_k(f_v: np.ndarray, f_prev: np.ndarray,
                 g1: float, g2: float, lam_v: float) -> np.ndarray:
        """Gradient of augmented Lagrangian w.r.t. f (includes proximal)."""
        obj_grad = _obj_grad_base(f_v)
        eq1_r = h * f_v.sum() - 1.0
        eq2_r = h * x.dot(f_v) - mean_target
        ineq_r = h * f_v.dot(f_v) - l2_budget
        aug    = max(0.0, lam_v + tau * ineq_r)
        return (obj_grad
                + (g1 + tau * eq1_r) * h
                + (g2 + tau * eq2_r) * h * x
                + aug * 2.0 * h * f_v
                + omega * (f_v - f_prev))

    def grad_outer(f_v: np.ndarray, g1: float, g2: float, lam_v: float) -> np.ndarray:
        """Gradient WITHOUT proximal (used for outer stationarity check)."""
        obj_grad = _obj_grad_base(f_v)
        eq1_r = h * f_v.sum() - 1.0
        eq2_r = h * x.dot(f_v) - mean_target
        ineq_r = h * f_v.dot(f_v) - l2_budget
        aug    = max(0.0, lam_v + tau * ineq_r)
        return (obj_grad
                + (g1 + tau * eq1_r) * h
                + (g2 + tau * eq2_r) * h * x
                + aug * 2.0 * h * f_v)

    # Initialise at f^0 = 1 (as specified by paper)
    f   = np.ones(n)
    g1  = 0.0
    g2  = 0.0
    lam = 0.0

    risk_trace, stat_trace = [], []
    outer_iter = 0

    for k in range(max_outer):
        f_prev = f.copy()

        # ── Inner sub-problem: minimise J_k over f >= 0 ──────────────────────
        # Use L-BFGS-B (bounded) for reliable convergence of the nonneg-
        # constrained smooth quadratic-like sub-problem.
        res = sp_minimize(
            J_k, f_prev,
            args=(f_prev, g1, g2, lam),
            jac=grad_J_k,
            method='L-BFGS-B',
            bounds=[(0.0, None)] * n,
            options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 5000},
        )
        f = np.maximum(res.x, 0.0)   # enforce nonnegativity numerically

        # ── Outer residuals ───────────────────────────────────────────────────
        eq1_val  = h * f.sum() - 1.0
        eq2_val  = h * x.dot(f) - mean_target
        ineq_val = h * f.dot(f) - l2_budget
        feas_res = abs(eq1_val) + abs(eq2_val) + max(0.0, ineq_val)
        cs_res   = abs(lam * ineq_val)

        # Unit-step projected-gradient stationarity (f >= 0, no proximal)
        g_out    = grad_outer(f, g1, g2, lam)
        stat_res = np.linalg.norm(f - np.maximum(f - g_out, 0.0))

        risk_val = compute_risk(f)
        risk_trace.append(float(risk_val))
        stat_trace.append(float(stat_res))
        outer_iter = k + 1

        if outer_iter % 500 == 0:
            print(f"  f-ALM(n={n}) k={outer_iter:5d}: risk={risk_val:.8f} "
                  f"h*f2={h*f.dot(f):.4f} lam={lam:.4f} "
                  f"feas={feas_res:.2e} cs={cs_res:.2e} stat={stat_res:.2e}")

        # ── Multiplier updates ────────────────────────────────────────────────
        g1  += tau * eq1_val
        g2  += tau * eq2_val
        lam  = max(0.0, lam + tau * ineq_val)

        if feas_res < outer_tol and cs_res < outer_tol and stat_res < outer_tol:
            break

    risk_final = compute_risk(f)
    print(f"  f-ALM(n={n}) done: k={outer_iter}, risk={risk_final:.8f}, stat={stat_trace[-1]:.2e}")

    return {
        'outer_iters': outer_iter,
        'risk':        float(risk_final),
        'stat':        float(stat_trace[-1]) if stat_trace else np.nan,
        'eq1_res':     float(abs(h * f.sum() - 1.0)),
        'eq2_res':     float(abs(h * x.dot(f) - mean_target)),
        'ineq_res':    float(max(0.0, h * f.dot(f) - l2_budget)),
        'feas_res':    float(abs(h * f.sum() - 1.0) + abs(h * x.dot(f) - mean_target) + max(0.0, h * f.dot(f) - l2_budget)),
        'cs_res':      float(abs(lam * (h * f.dot(f) - l2_budget))),
        'risk_trace':  [float(r) for r in risk_trace],
        'stat_trace':  [float(s) for s in stat_trace],
        'f_opt':       f.tolist(),
    }


# ── CVXPY direct solve (sanity check) ────────────────────────────────────────

def cvxpy_quantile_space(n=80, mean_target=0.0, var_budget=1.0):
    if not CVXPY_AVAILABLE:
        return None
    h = 1.0 / n
    u = (np.arange(n) + 0.5) / n
    phi = 2.0 * u
    Q = cp.Variable(n)
    mono = [Q[i] <= Q[i + 1] for i in range(n - 1)]
    prob = cp.Problem(
        cp.Maximize(h * (phi @ Q)),
        [h * cp.sum(Q) == mean_target,
         h * cp.sum_squares(Q) <= var_budget] + mono
    )
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-7)
    if prob.status in ('optimal', 'optimal_inaccurate'):
        return float(prob.value)
    return None


def cvxpy_density_space(n: int = 80, mean_target: float = 0.5, l2_budget: float = 2.0):
    """
    Exact density-space problem via CVXPY.
    Minimize h*sum(F^2)  [= maximize 1-h*sum(F^2) = R_phi(F)]
    subject to normalization, mean, L2 budget h*sum(f^2)<=l2_budget, f>=0.
    """
    if not CVXPY_AVAILABLE:
        return None
    h = 1.0 / n
    x = (np.arange(n) + 0.5) / n
    # CDF matrix: F = h * L @ f
    L = np.zeros((n, n))
    for i in range(n):
        L[i, :i] = 1.0
        L[i,  i] = 0.5
    f_var = cp.Variable(n, nonneg=True)
    F_var = h * (L @ f_var)
    prob = cp.Problem(
        cp.Minimize(h * cp.sum_squares(F_var)),
        [h * cp.sum(f_var) == 1.0,
         h * (x @ f_var) == mean_target,
         h * cp.sum_squares(f_var) <= l2_budget]   # correct L2 budget
    )
    prob.solve(solver=cp.SCS, verbose=False)
    if prob.status in ('optimal', 'optimal_inaccurate') and f_var.value is not None:
        return float(1.0 - prob.value)   # R_phi = 1 - h*sum(F^2)
    return None


# ── Convergence plots ─────────────────────────────────────────────────────────

def plot_convergence(results_dict, label, fname):
    traces = {k: v for k, v in results_dict.items() if k.endswith('_trace')}
    n_panels = len(traces)
    if n_panels == 0:
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    for ax, (name, trace) in zip(axes, traces.items()):
        ax.semilogy(trace, 'b-', linewidth=1)
        ax.set_xlabel('Outer iteration')
        ax.set_ylabel(name.replace('_trace', ''))
        ax.set_title(f'{label}: {name}')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

TABLE2_TARGETS = {
    'quantile_n80': {'iters': 60, 'risk': 0.57730534, 'stat': 2.94e-7},
    'density_n80':  {'iters': 261, 'risk': 0.71870715, 'stat': 2.99e-7},
    'density_n160': {'iters': 261, 'risk': 0.71870693, 'stat': 2.98e-7},
}


def main():
    results_all = {}

    # ── Quantile space n=80 ──
    print("=" * 60)
    print("Quantile-space ALM  (n=80, phi=2u, mean=0, var<=1)")
    print("=" * 60)
    r_q80 = quantile_space_ALM(n=80)
    tgt = TABLE2_TARGETS['quantile_n80']
    print(f"  outer iters : {r_q80['outer_iters']:4d}   target ~{tgt['iters']}")
    print(f"  risk        : {r_q80['risk']:.8f}   target ~{tgt['risk']:.8f}")
    print(f"  stat        : {r_q80['stat']:.2e}   target ~{tgt['stat']:.2e}")
    print(f"  feas_res    : {r_q80['feas_res']:.2e}")
    print(f"  cs_res      : {r_q80['cs_res']:.2e}")
    results_all['quantile_n80'] = r_q80

    cvx_q = cvxpy_quantile_space(n=80)
    if cvx_q is not None:
        print(f"  CVXPY sanity: {cvx_q:.8f}   diff={abs(cvx_q - r_q80['risk']):.2e}")
        results_all['quantile_n80_cvxpy'] = cvx_q

    plot_convergence(r_q80, 'Quantile ALM (n=80)',
                     '/workspace/project/results/exp3_convergence_quantile_n80.png')
    print("  Saved convergence plot.")

    # ── Density space n=80 ──
    print("\n" + "=" * 60)
    print("Density-space ALM  (n=80, phi=2u, mean=1/2, L2 budget=2)")
    print("=" * 60)
    r_d80 = density_space_ALM(n=80)
    tgt = TABLE2_TARGETS['density_n80']
    print(f"  outer iters : {r_d80['outer_iters']:4d}   target ~{tgt['iters']}")
    print(f"  risk        : {r_d80['risk']:.8f}   target ~{tgt['risk']:.8f}")
    print(f"  stat        : {r_d80['stat']:.2e}   target ~{tgt['stat']:.2e}")
    print(f"  eq1_res     : {r_d80['eq1_res']:.2e}")
    print(f"  eq2_res     : {r_d80['eq2_res']:.2e}")
    results_all['density_n80'] = r_d80

    cvx_d80 = cvxpy_density_space(n=80)
    if cvx_d80 is not None:
        print(f"  CVXPY sanity: {cvx_d80:.8f}   diff={abs(cvx_d80 - r_d80['risk']):.2e}")
        results_all['density_n80_cvxpy'] = cvx_d80

    plot_convergence(r_d80, 'Density ALM (n=80)',
                     '/workspace/project/results/exp3_convergence_density_n80.png')
    print("  Saved convergence plot.")

    # ── Density space n=160 ──
    print("\n" + "=" * 60)
    print("Density-space ALM  (n=160, phi=2u, mean=1/2, L2 budget=2)")
    print("=" * 60)
    r_d160 = density_space_ALM(n=160)
    tgt = TABLE2_TARGETS['density_n160']
    print(f"  outer iters : {r_d160['outer_iters']:4d}   target ~{tgt['iters']}")
    print(f"  risk        : {r_d160['risk']:.8f}   target ~{tgt['risk']:.8f}")
    print(f"  stat        : {r_d160['stat']:.2e}   target ~{tgt['stat']:.2e}")
    results_all['density_n160'] = r_d160

    plot_convergence(r_d160, 'Density ALM (n=160)',
                     '/workspace/project/results/exp3_convergence_density_n160.png')
    print("  Saved convergence plot.")

    # ── CSV table ──
    rows = []
    for key in ('quantile_n80', 'density_n80', 'density_n160'):
        r = results_all[key]
        t = TABLE2_TARGETS.get(key, {})
        rows.append({
            'method': key,
            'outer_iters': r['outer_iters'],
            'risk': r['risk'],
            'stat': r['stat'],
            'target_iters': t.get('iters', ''),
            'target_risk': t.get('risk', ''),
            'target_stat': t.get('stat', ''),
        })

    with open('/workspace/project/results/exp3_table2.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved exp3_table2.csv")

    # Strip large arrays before saving JSON
    for key in ('quantile_n80', 'density_n80', 'density_n160'):
        r = results_all[key]
        results_all[f'{key}_risk_trace'] = r.pop('risk_trace', [])
        results_all[f'{key}_stat_trace'] = r.pop('stat_trace', [])
        r.pop('Q_opt', None)
        r.pop('f_opt', None)

    with open('/workspace/project/results/exp3_results.json', 'w') as f:
        json.dump(results_all, f, indent=2, default=str)
    print("Saved exp3_results.json")


if __name__ == '__main__':
    main()
