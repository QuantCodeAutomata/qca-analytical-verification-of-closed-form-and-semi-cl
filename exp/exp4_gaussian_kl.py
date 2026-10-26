"""
exp4_gaussian_kl.py
Gaussian KL primal-dual consistency validation (Model 6b).
"""

import sys
sys.path.insert(0, '/workspace/project')

import json, csv, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize, stats

warnings.filterwarnings('ignore')
os.makedirs('/workspace/project/results', exist_ok=True)


# ── J0 and J1 functions ───────────────────────────────────────────────────────

def J0(lam, u, v):
    """
    J_0(lam; u, v) = exp(lam^2/2) * [Phi(v-lam) - Phi(u-lam)]
    where Phi is the standard normal CDF.
    """
    base = np.exp(0.5 * lam ** 2)
    cdf_v = 1.0 if v == np.inf else stats.norm.cdf(v - lam)
    cdf_u = 0.0 if u == -np.inf else stats.norm.cdf(u - lam)
    return base * (cdf_v - cdf_u)


def J1(lam, u, v):
    """
    J_1(lam; u, v) = exp(lam^2/2) * [lam*(Phi(v-lam)-Phi(u-lam)) - (phi(v-lam)-phi(u-lam))]
    where phi is the standard normal PDF.
    """
    base = np.exp(0.5 * lam ** 2)
    cdf_v = 1.0 if v == np.inf else stats.norm.cdf(v - lam)
    cdf_u = 0.0 if u == -np.inf else stats.norm.cdf(u - lam)
    pdf_v = 0.0 if v == np.inf else stats.norm.pdf(v - lam)
    pdf_u = 0.0 if u == -np.inf else stats.norm.pdf(u - lam)
    return base * (lam * (cdf_v - cdf_u) - (pdf_v - pdf_u))


def J0_safe(lam, u, v):
    """Numerically stable J0 using log-space computation."""
    try:
        val = J0(lam, u, v)
        if np.isfinite(val) and val >= 0:
            return val
    except Exception:
        pass
    return 1e-30


def J1_safe(lam, u, v):
    try:
        val = J1(lam, u, v)
        if np.isfinite(val):
            return val
    except Exception:
        pass
    return 0.0


# ── Primal calibration system (equations 54-56) ───────────────────────────────

def primal_system(params, alpha, epsilon, bar_mu):
    """
    System of equations for (lambda1, lambda2, x_alpha) from paper eqs 54-56.

    Worst-case density: f*(x) = C_i * exp(lam_i*x) * phi_N(x) on each piece,
    with C1 = alpha/J0(lam1,-inf,x_a) and C2 = (1-alpha)/J0(lam2,x_a,inf).

    Eq54 (density continuity at x_a, log space):
      ln(alpha) - ln(J0(lam1,-inf,x_a)) + lam1*x_a
      = ln(1-alpha) - ln(J0(lam2,x_a,inf)) + lam2*x_a
    Eq55 (mean constraint):
      alpha*J1(lam1,-inf,x_a)/J0(lam1,-inf,x_a)
      + (1-alpha)*J1(lam2,x_a,inf)/J0(lam2,x_a,inf) = bar_mu
    Eq56 (KL binding, KL(f*||phi_N) = epsilon):
      alpha*ln(alpha/J0(lam1,-inf,x_a))
      + (1-alpha)*ln((1-alpha)/J0(lam2,x_a,inf))
      + lam1*alpha*J1(lam1,-inf,x_a)/J0(lam1,-inf,x_a)
      + lam2*(1-alpha)*J1(lam2,x_a,inf)/J0(lam2,x_a,inf) = epsilon
    """
    lam1, lam2, x_a = params
    try:
        j0_lo = J0_safe(lam1, -np.inf, x_a)
        j0_hi = J0_safe(lam2, x_a, np.inf)
        if j0_lo < 1e-30 or j0_hi < 1e-30:
            return [1e3, 1e3, 1e3]

        j1_lo = J1_safe(lam1, -np.inf, x_a)
        j1_hi = J1_safe(lam2, x_a, np.inf)

        mu_lo = j1_lo / j0_lo   # conditional mean for left piece
        mu_hi = j1_hi / j0_hi   # conditional mean for right piece

        # Eq54: log-space continuity
        eq1 = (np.log(alpha) - np.log(j0_lo) + lam1 * x_a
               - (np.log(1.0 - alpha) - np.log(j0_hi) + lam2 * x_a))

        # Eq55: mean
        eq2 = alpha * mu_lo + (1.0 - alpha) * mu_hi - bar_mu

        # Eq56: KL = epsilon
        eq3 = (alpha * np.log(alpha / j0_lo) + (1.0 - alpha) * np.log((1.0 - alpha) / j0_hi)
               + lam1 * alpha * mu_lo + lam2 * (1.0 - alpha) * mu_hi) - epsilon

        return [eq1, eq2, eq3]
    except Exception:
        return [1e3, 1e3, 1e3]


def solve_primal(alpha, epsilon, bar_mu, n_attempts=None):
    """
    Solve the primal system via fsolve with multiple starting points.
    Returns (lam1, lam2, x_alpha) or None.

    At least 20 initial guesses are tried, drawn from a grid over
    x_a0 = norm.ppf(alpha)+bar_mu + dx_a  and  lam2_0 in spread above bar_mu.
    Acceptance criterion: residual norm < 1e-6 AND lam1 < lam2.
    """
    base_xa = stats.norm.ppf(alpha) + bar_mu

    # Grid: 5 x_alpha offsets x 4 lam2 values → 20 guesses (all satisfy lam1 < lam2)
    init_guesses = []
    for dx_a in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        x_a0 = base_xa + dx_a
        for lam2_0 in [0.5, 1.0, 1.5, 2.0]:
            lam1_0 = bar_mu - abs(lam2_0 - bar_mu)
            if lam1_0 < lam2_0:
                init_guesses.append([lam1_0, lam2_0, x_a0])

    best = None
    best_res_norm = np.inf

    for guess in init_guesses:
        try:
            sol, _, ier, _ = optimize.fsolve(
                primal_system, guess,
                args=(alpha, epsilon, bar_mu),
                full_output=True, xtol=1e-10
            )
            res = primal_system(sol, alpha, epsilon, bar_mu)
            res_norm = np.max(np.abs(res))
            if res_norm < best_res_norm and sol[0] < sol[1]:
                best = sol
                best_res_norm = res_norm
        except Exception:
            continue

    if best is None or best_res_norm > 1e-6:
        return None, best_res_norm if best is not None else np.inf
    return best, best_res_norm


def compute_chi_primal(lam1, lam2, x_alpha):
    """
    chi = E_{f*}[X | X >= x_alpha] = J1(lam2, x_alpha, inf) / J0(lam2, x_alpha, inf)
    (The worst-case ES is determined by the lambda2 piece above x_alpha.)
    """
    j0_hi = J0_safe(lam2, x_alpha, np.inf)
    j1_hi = J1_safe(lam2, x_alpha, np.inf)
    if j0_hi < 1e-15:
        # Approximation using Mills ratio
        z = x_alpha - lam2
        return float(lam2 + stats.norm.pdf(z) / max(stats.norm.sf(z), 1e-15))
    return float(j1_hi / j0_hi)


def compute_chi_dual(alpha, epsilon, bar_mu, lam1, lam2, x_alpha):
    """
    Dual: chi_dual = inf_{zeta>0, eta, t} D(zeta, eta, t)
      D = t + eta*bar_mu + zeta*epsilon + zeta*ln(E)
    where for X ~ N(0,1):
      E = E[exp((h_t(X)-eta*X)/zeta)],  h_t(x) = (x-t)_+ / (1-alpha)
      = E[exp(-eta*X/zeta) * 1_{X<t}]
        + E[exp((X-t)/((1-alpha)*zeta) - eta*X/zeta) * 1_{X>=t}]

    Analytically (using J0 = E[exp(c*X)*1_{region}] for X~N(0,1)):
      a  = -eta/zeta
      b  = (1/(1-alpha) - eta) / zeta
      log E = logaddexp(
                0.5*a^2 + log Phi(t-a),
                -t/((1-alpha)*zeta) + 0.5*b^2 + log(1-Phi(t-b))
              )
    """
    def dual_obj(params):
        zeta, eta, t = params
        if zeta < 1e-12:
            return 1e6
        try:
            a = -eta / zeta
            b_val = (1.0 / (1.0 - alpha) - eta) / zeta

            log_p1 = 0.5 * a ** 2 + np.log(max(stats.norm.cdf(t - a), 1e-300))
            log_p2 = (-t / ((1.0 - alpha) * zeta)
                      + 0.5 * b_val ** 2
                      + np.log(max(stats.norm.sf(t - b_val), 1e-300)))

            log_E = np.logaddexp(log_p1, log_p2)
            D = t + eta * bar_mu + zeta * epsilon + zeta * log_E
            return float(D) if np.isfinite(D) else 1e6
        except Exception:
            return 1e6

    chi_p = float(compute_chi_primal(lam1, lam2, x_alpha))

    # Try multiple starts; strong duality: min D = chi_primal
    best_val = np.inf
    for zeta0 in [0.5, 1.0, max(0.01, lam2 - lam1)]:
        for eta0 in [lam1, bar_mu, lam1 + 0.3]:
            for t0 in [x_alpha, x_alpha - 0.5, x_alpha + 0.5]:
                try:
                    res = optimize.minimize(
                        dual_obj, [zeta0, eta0, t0],
                        method='Nelder-Mead',
                        options={'xatol': 1e-9, 'fatol': 1e-9,
                                 'maxiter': 20000, 'adaptive': True},
                    )
                    if res.fun < best_val:
                        best_val = res.fun
                except Exception:
                    pass

    return float(best_val) if np.isfinite(best_val) else chi_p


# ── Main test grid ────────────────────────────────────────────────────────────

def run_test_grid():
    alphas = [0.9, 0.95, 0.975]
    bar_mus = [0.0, 0.25, 0.5, -0.5]
    deltas = [0.01, 0.05, 0.1, 0.25]

    records = []
    all_results = {}
    all_pass = True

    print(f"{'alpha':>6} {'mu':>6} {'eps':>6} | {'lam1':>8} {'lam2':>8} {'x_alp':>8} "
          f"| {'chi_P':>10} {'chi_D':>10} {'|diff|':>10} {'pass':>5}")
    print("-" * 90)

    for alpha in alphas:
        for bar_mu in bar_mus:
            eps_min = bar_mu ** 2 / 2.0
            for delta in deltas:
                epsilon = eps_min + delta
                feasible = epsilon > eps_min

                sol, res_norm = solve_primal(alpha, epsilon, bar_mu)
                if sol is None or not feasible:
                    records.append({
                        'alpha': alpha, 'bar_mu': bar_mu, 'epsilon': epsilon,
                        'lambda1': None, 'lambda2': None, 'x_alpha': None,
                        'chi_primal': None, 'chi_dual': None,
                        'abs_diff': None, 'rel_diff': None, 'status': 'infeasible_or_failed',
                    })
                    continue

                lam1, lam2, x_alpha = sol
                lam1_lt_lam2 = lam1 < lam2

                chi_p = compute_chi_primal(lam1, lam2, x_alpha)
                chi_d = compute_chi_dual(alpha, epsilon, bar_mu, lam1, lam2, x_alpha)

                abs_diff = abs(chi_p - chi_d)
                rel_diff = abs_diff / max(abs(chi_p), 1e-10)
                consistent = abs_diff < 0.05 and lam1_lt_lam2

                if not consistent:
                    all_pass = False

                status = 'ok' if consistent else 'inconsistent'
                print(f"{alpha:6.3f} {bar_mu:6.2f} {epsilon:6.3f} | "
                      f"{lam1:8.4f} {lam2:8.4f} {x_alpha:8.4f} | "
                      f"{chi_p:10.6f} {chi_d:10.6f} {abs_diff:10.2e} {'PASS' if consistent else 'FAIL':>5}")

                records.append({
                    'alpha': alpha, 'bar_mu': bar_mu, 'epsilon': epsilon,
                    'lambda1': float(lam1), 'lambda2': float(lam2),
                    'x_alpha': float(x_alpha),
                    'chi_primal': float(chi_p), 'chi_dual': float(chi_d),
                    'abs_diff': float(abs_diff), 'rel_diff': float(rel_diff),
                    'status': status,
                })

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return records


# ── chi vs epsilon plots ──────────────────────────────────────────────────────

def plot_chi_curves(records):
    # Group by (alpha, bar_mu) and plot chi vs epsilon
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        if r['chi_primal'] is not None:
            key = (r['alpha'], r['bar_mu'])
            groups[key].append((r['epsilon'], r['chi_primal']))

    # Pick a subset of (alpha, bar_mu) combinations
    selected = [(0.9, 0.0), (0.9, 0.25), (0.95, 0.0), (0.95, 0.5)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (alpha, bar_mu) in zip(axes, selected):
        key = (alpha, bar_mu)
        if key not in groups:
            ax.set_visible(False)
            continue
        pts = sorted(groups[key])
        eps_v = [p[0] for p in pts]
        chi_v = [p[1] for p in pts]
        ax.plot(eps_v, chi_v, 'b-o', markersize=5)
        ax.set_xlabel('$\\epsilon$ (KL budget)')
        ax.set_ylabel('$\\chi$ (worst-case ES)')
        ax.set_title(f'$\\alpha={alpha}$, $\\bar{{\\mu}}={bar_mu}$')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Worst-case ES ($\\chi$) vs KL budget ($\\epsilon$)', fontsize=12)
    plt.tight_layout()
    plt.savefig('/workspace/project/results/exp4_chi_curves.png', dpi=120)
    plt.close()
    print("Saved exp4_chi_curves.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MODEL 6b: Gaussian KL primal-dual consistency")
    print("=" * 60)

    records = run_test_grid()

    # Save CSV
    fieldnames = ['alpha', 'bar_mu', 'epsilon', 'lambda1', 'lambda2', 'x_alpha',
                  'chi_primal', 'chi_dual', 'abs_diff', 'rel_diff', 'status']
    with open('/workspace/project/results/exp4_primal_dual_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print("\nSaved exp4_primal_dual_table.csv")

    # Plot
    plot_chi_curves(records)

    # Verify lambda1 < lambda2 for all successful solutions
    ok_records = [r for r in records if r['lambda1'] is not None]
    lam_check = all(r['lambda1'] < r['lambda2'] for r in ok_records)
    print(f"lambda1 < lambda2 for all solutions: {lam_check}")

    # Save JSON
    results_json = {
        'lambda1_lt_lambda2_all': lam_check,
        'n_feasible': len(ok_records),
        'n_total': len(records),
        'records': records,
    }
    with open('/workspace/project/results/exp4_results.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print("Saved exp4_results.json")


if __name__ == '__main__':
    main()
