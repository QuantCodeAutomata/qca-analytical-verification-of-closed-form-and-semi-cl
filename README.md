# Analytical Verification of Closed-Form and Semi-Closed-Form Worst-Case Spectral Risk Laws

A research repository implementing and numerically validating theorem-level analytical constructions for **worst-case spectral risk measures** under six parametric distributional ambiguity models, together with projection-based proximal augmented-Lagrangian algorithms and a portfolio reduction theorem.

---

## Structure

```
.
├── data/               # Data notes (no external market data required)
├── src/                # Core model and algorithm implementations
│   ├── spectral_risk.py          # Spectral risk measure primitives
│   ├── model1_central_moment.py  # Model 1: central-moment ambiguity
│   ├── model2_wasserstein_es.py  # Model 2: Wasserstein + ES stress
│   ├── model3_mean_variance_es.py# Model 3: mean-variance + two-sided ES band
│   ├── model4_partial_moments.py # Model 4: exact partial moments (bounded support)
│   ├── model5_geometric_moment.py# Model 5: mean + geometric-moment bound
│   ├── model6a_kl_exponential.py # Model 6a: KL vs exponential reference
│   ├── model6b_kl_gaussian.py    # Model 6b: KL vs Gaussian reference
│   ├── algorithms.py             # Projection-based proximal ALM (exp3)
│   └── portfolio_reduction.py    # Gaussian KL portfolio SOCP reduction (exp5)
├── exp/                # Experiment scripts
│   ├── exp1_analytical_verification.py  # Theorems 1-8 analytical verification
│   ├── exp2_figures.py                  # Figures 1-4 reproduction
│   ├── exp3_algorithms.py               # Table 2: ALM convergence
│   ├── exp4_gaussian_kl.py              # Gaussian KL primal–dual consistency
│   └── exp5_portfolio.py                # Portfolio projection identity & SOCP
├── tests/              # pytest unit and integration tests (42 tests)
├── results/            # Figures, CSV outputs, JSON diagnostics, RESULTS.md
└── RESULTS.md → results/RESULTS.md
```

---

## Experiments

### Exp 1 — Analytical Verification (Theorems 1–8)
Implements and validates all six ambiguity models:
- **Model 1** Central-moment ambiguity with kappa-th moment budget
- **Model 2** Wasserstein L2 ambiguity with ES stress floor
- **Model 3** Mean-variance + two-sided ES band (benchmark / floor-active / cap-active regimes)
- **Model 4** Exact partial moments on bounded support (inverse-spectrum construction)
- **Model 5** Mean + geometric-moment bound (dilogarithm calibration equation)
- **Model 6a** KL ambiguity against exponential reference (piecewise-exponential density)
- **Model 6b** KL ambiguity against Gaussian reference (exponentially tilted Gaussian)

Each model is cross-checked against an independent CVXPY discretised convex optimiser.

### Exp 2 — Figures 1–4 Reproduction
Reproduces the paper's main figures with parameter sweeps, regime-transition markers, and
scalar benchmarks:
- Figure 1: Wasserstein + ES regime switch at c_dagger ≈ 4.0087 (target 4.0094)
- Figure 2: Mean-variance ES benchmark bar_c ≈ 22.118 (target 22.117)
- Figure 3: Partial-moment worst-case CDF curves over asymmetry t
- Figure 4: KL exponential piecewise density and ES vs epsilon curve

### Exp 3 — Projection-Based ALM (Table 2 Reproduction)
Proximal augmented-Lagrangian with projected-gradient inner loop on fixed grids:
- **Quantile-space test** (n=80): risk 0.57730541 vs paper 0.57730534
- **Density-space test** (n=80): risk 0.71870530 vs paper 0.71870715
- **Density-space test** (n=160): risk 0.71859182 vs paper 0.71870693
- CVXPY cross-check confirms ALM finds the correct optima.

### Exp 4 — Gaussian KL Primal–Dual Consistency
- Solves equations (54)–(56) for 44 (alpha, bar_mu, epsilon) triples.
- **All 44 scenarios pass**: chi_primal = chi_dual to machine precision (< 1e-13 absolute diff).
- Structural inequality lambda_1 < lambda_2 holds universally.

### Exp 5 — Portfolio Projection Identity and SOCP Reduction
- Verifies V(w) = mu_w + sigma_w * chi(alpha, epsilon) for multiple portfolios and dimensions.
- Projection identity error < 1e-14 in all tested cases.
- SOCP reduction solved with cvxpy; chi is confirmed dimension-independent.

---

## Running

```bash
# Install dependencies
pip install numpy scipy matplotlib pandas mpmath cvxpy

# Run all experiments
python exp/exp1_analytical_verification.py
python exp/exp2_figures.py
python exp/exp3_algorithms.py
python exp/exp4_gaussian_kl.py
python exp/exp5_portfolio.py

# Run tests
pytest tests/ -v
```

All outputs are saved to `results/`. See `results/RESULTS.md` for a full summary.

---

## Key Dependencies
- `numpy`, `scipy` — numerics and root finding
- `mpmath` — dilogarithm (Li_2) for Model 5
- `cvxpy` — SOCP portfolio optimisation and numerical cross-checks
- `matplotlib` — all figures
- `pandas` — tabular output

No external market data (Massive API or otherwise) is required — all inputs are synthetic/parametric.
