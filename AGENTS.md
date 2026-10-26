# Repository knowledge for OpenHands agents

## Project overview
Distributional-robustness / spectral risk measure library.
All code lives in `src/`; experiments in `exp/`; tests in `tests/`.
No historical data required — everything is synthetic/parametric.

## Source modules

| File | Purpose |
|------|---------|
| `src/spectral_risk.py` | Core primitives: `SpectralRiskMeasure`, `phi_power`, `phi_exp`, `Phi_func`, `risk_quantile_space`, `risk_cdf_space` |
| `src/model1_central_moment.py` | Model 1 — central-moment ambiguity; solver for gamma1, lambda1, Q* |
| `src/model2_wasserstein_es.py` | Model 2 — Wasserstein ball + ES floor; three regimes (slack/active/infeasible) |
| `src/model3_mean_variance_es.py` | Model 3 — mean-variance + two-sided ES band; floor/cap/benchmark regimes |
| `src/model4_partial_moments.py` | Model 4 — exact partial moments on bounded support; piecewise power-law CDF |
| `src/model5_geometric_moment.py` | Model 5 — geometric-moment bound; uses `mpmath.polylog`, exp spectrum |
| `src/model6a_kl_exponential.py` | Model 6a — KL + exponential reference; piecewise exponential worst-case density |
| `src/model6b_kl_gaussian.py` | Model 6b — KL + Gaussian reference; piecewise tilted-Gaussian; chi table |
| `src/algorithms.py` | `weighted_PAV`, `QuantileSpaceALM`, `DensitySpaceALM` — projected-gradient ALM |
| `src/portfolio_reduction.py` | Portfolio SOCP reduction via Model 6b projection identity |

## Key mathematical conventions
- Quantile function Q(u) is left-continuous, defined on [0,1].
- Spectral risk: R_phi(Q) = ∫₀¹ phi(u) Q(u) du.
- phi integrates to 1 and is non-decreasing.
- Model 1 gamma1 satisfies ∫|phi-gamma|^{1/(κ-1)} sgn(phi-gamma) du = 0.
- Model 2 three regimes: `c <= c_G` → slack; `c_G < c <= c_dagger` → active; else infeasible.
- Model 6b chi: `compute_chi_primal(lambda2, x_alpha)` = lambda2 + phi_N(x_a-lam2)/(1-Phi_N(x_a-lam2)).
- Portfolio identity: worst-case ES = mu_w + sigma_w * chi(alpha, epsilon).

## Dependencies
`numpy`, `scipy`, `pandas`, `cvxpy` (cross-checks), `mpmath` (optional, Model 5 polylog).

## Experiment scripts (exp/)
All five scripts are self-contained; they handle `ImportError` from `src/` gracefully
with inline fallback implementations and `sys.path.insert(0, '/workspace/project')`.

| Script | Purpose |
|--------|---------|
| `exp1_analytical_verification.py` | Runs analytical cross-checks for Models 1–6b; saves `results/exp1_results.json` |
| `exp2_figures.py` | Reproduces paper Figures 1–4; saves PNG + CSV in `results/` |
| `exp3_algorithms.py` | ALM convergence tests vs Table 2 targets; saves `results/exp3_*.{csv,json}` |
| `exp4_gaussian_kl.py` | Primal/dual consistency sweep over (alpha, bar_mu, epsilon); saves `results/exp4_*.{csv,json}` |
| `exp5_portfolio.py` | Portfolio SOCP validation; saves `results/exp5_*.{csv,json}` |

## Test suite (tests/) — 42 items, all pass
Run with: `python -m pytest tests/ -v`

Key tolerance notes:
- Grids over `linspace(0.001, 0.999, n)` miss the boundary singularity of Q_G(u)=-ln(1-u) near u=1;
  tolerances in `test_cG_value` and `test_c_dagger_value` are set to 0.1 (not 0.05) for this reason.
- `phi = 2u` integrates to 0.98 over [0.01, 0.99]; `test_power_spectrum_beta2` uses tolerance 0.025.
- `J0(lam, -inf, inf) = exp(lam^2/2)` (moment-generating function), NOT 1.
  `test_J0_positive` asserts `abs(J0 - exp(lam^2/2)) < 0.001`, not equality to 1.

## Key numerical facts
- `c_G = 1 - ln(0.1) ≈ 3.302585` for Q_G(u) = -ln(1-u), alpha=0.9
- `c_dagger ≈ 4.0094` for epsilon=0.35, phi=3u^2, alpha=0.9
- `bar_c ≈ 22.117` for mu=3, sigma=10, alpha=0.9, beta=3 power spectrum
- `M_alpha = mu + sigma * sqrt(alpha/(1-alpha))` for ES upper bound
- Model 6b chi_primal = J1(lam2, x_alpha, inf) / J0(lam2, x_alpha, inf)
  (identical to `lam2 + phi_N(x_alpha-lam2) / Phi_N^c(x_alpha-lam2)`)
- Model 6b dual strong-duality: min_D(zeta,eta,t) = chi_primal (Nelder-Mead from primal point)

## Model 6a (KL + Exponential reference) — corrected design
- **KL reference is the FULL exponential** g(x)=theta*exp(-theta*x) on [0,∞), NOT the truncated-renormalized density on [0,b]. No Z=1-exp(-theta*b) denominator in KL.
- `_kl_piecewise`: KL_left = alpha*ln(C1/theta) + C1*(theta-xi1)*I1(xi1,0,x_a); KL_right analogous.
- `I0`/`I1` stability threshold: `abs(xi) < 1e-6` (was too tight; caused cancellation at small xi).
- `solve_system_6a` uses **multi-start** with 13 guesses spanning x_alpha in [0.4,0.8]*ref_quantile; also calls `least_squares` with bounds alongside `fsolve` for each start.
- `compute_epsilon_min_6a` uses the feasibility lemma: find varsigma* s.t. I1(varsigma,0,b)/I0(varsigma,0,b)=mu via brentq, then epsilon_min = ln(C(varsigma*))-ln(theta)+(theta-varsigma*)*mu.
- Paper targets (theta=1, b=5, alpha=0.95, mu=0.5, eps=0.2): xi1≈2.1885, xi2≈1.3360, x_alpha≈1.5808, epsilon_min≈0.193193.

## exp4_gaussian_kl.py (Gaussian KL primal-dual, Model 6b)
- **Correct primal system uses equations 54-56** (continuity condition, NOT quantile condition):
  - Eq54 (continuity, log-space): ln(alpha)-ln(J0_lo)+lam1*xa = ln(1-alpha)-ln(J0_hi)+lam2*xa
  - Eq55 (mean): alpha*J1_lo/J0_lo + (1-alpha)*J1_hi/J0_hi = bar_mu
  - Eq56 (KL binding): alpha*ln(alpha/J0_lo)+(1-alpha)*ln((1-alpha)/J0_hi)+lam1*alpha*J1_lo/J0_lo+lam2*(1-alpha)*J1_hi/J0_hi = epsilon
- The continuity condition (eq54) enforces the KKT optimality condition; it gives the TRUE worst-case ES, which is HIGHER than the value from the simpler quantile-only formulation used in exp5.
- exp5's `calibrate_chi` uses a different (non-continuity) formulation and finds a feasible but suboptimal solution. For alpha=0.9, eps=0.1, bar_mu=0: eq54-56 gives chi≈2.584; exp5 gives chi≈2.460.
- Primal-dual consistency is confirmed: dual min D(zeta,eta,t) = chi_primal to machine precision.
- `solve_primal` uses 20 starts: 5 x_alpha offsets × 4 lam2 values, all with lam1 < lam2.
- `compute_chi_dual` analytically evaluates E[exp((h_t(X)-eta*X)/zeta)] via logaddexp of two Gaussian integrals (a=-eta/zeta, b=(1/(1-alpha)-eta)/zeta terms).
- `fsolve` does NOT accept `ftol`; only `xtol` is valid for fsolve (ftol is for least_squares).

## Pitfalls
- Model 4 `solve_F_mu` uses brentq on the first partial moment residual; the second partial
  moment `s2` is used only in `compute_multipliers_model4` (not in the root-finding loop)
  — this is a deliberate simplification (noted OCR-issue provenance).
- Model 5 `compute_risk_model5` requires x_L > 0 and x_U > x_L; callers should check.
- `DensitySpaceALM._gradient` uses finite-difference (slow for large grids).
- Model 6b dual_obj uses log-sum-exp for overflow safety (`np.logaddexp`); guard `shift > 500`.
