"""
spectral_risk.py
----------------
Core spectral risk-measure primitives.

A spectral risk measure is characterised by a non-negative, non-decreasing
weight function phi : [0,1] -> R_+ with integral_0^1 phi(u) du = 1 and is
defined by

    R_phi(X)  =  integral_0^1  phi(u) * Q_X(u) du

where Q_X is the left-continuous quantile function (generalised inverse CDF)
of the loss X.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Spectrum families
# ---------------------------------------------------------------------------

def phi_power(u: float | NDArray, beta: float) -> float | NDArray:
    """Power-law spectrum  phi(u) = beta * u^{beta-1}.

    Parameters
    ----------
    u:
        Probability level(s) in [0, 1].
    beta:
        Shape parameter, beta > 1 guarantees the spectrum is non-decreasing.

    Returns
    -------
    float | NDArray
        Spectrum value(s).
    """
    u = np.asarray(u, dtype=float)
    return beta * u ** (beta - 1.0)


def phi_exp(u: float | NDArray, nu: float) -> float | NDArray:
    """Exponential spectrum  phi(u) = nu * exp(nu*u) / (exp(nu) - 1).

    Parameters
    ----------
    u:
        Probability level(s) in [0, 1].
    nu:
        Concentration parameter, nu > 0.

    Returns
    -------
    float | NDArray
        Spectrum value(s).
    """
    u = np.asarray(u, dtype=float)
    return nu * np.exp(nu * u) / (np.exp(nu) - 1.0)


# ---------------------------------------------------------------------------
# Cumulative spectrum  Phi(t) = integral_0^t phi(v) dv
# ---------------------------------------------------------------------------

def Phi_func(
    t: float,
    phi_callable: Callable[..., float],
    **kwargs,
) -> float:
    """Compute the cumulative spectrum  Phi(t) = integral_0^t phi(v) dv.

    Parameters
    ----------
    t:
        Upper limit in [0, 1].
    phi_callable:
        Spectrum function with signature phi(u, **kwargs).
    **kwargs:
        Additional keyword arguments forwarded to *phi_callable*.

    Returns
    -------
    float
        Phi(t).
    """
    if t <= 0.0:
        return 0.0
    result, _ = quad(lambda v: phi_callable(v, **kwargs), 0.0, t)
    return float(result)


# ---------------------------------------------------------------------------
# Discrete quadrature helpers
# ---------------------------------------------------------------------------

def risk_quantile_space(
    phi_vals: NDArray,
    Q_vals: NDArray,
    u_grid: NDArray,
) -> float:
    """Approximate R_phi(Q) via trapezoidal quadrature in quantile space.

    Parameters
    ----------
    phi_vals:
        Spectrum evaluated on *u_grid*, shape (n,).
    Q_vals:
        Quantile function evaluated on *u_grid*, shape (n,).
    u_grid:
        Uniform or non-uniform grid in [0, 1], shape (n,).

    Returns
    -------
    float
        Numerical approximation of integral_0^1 phi(u) Q(u) du.
    """
    phi_vals = np.asarray(phi_vals, dtype=float)
    Q_vals = np.asarray(Q_vals, dtype=float)
    u_grid = np.asarray(u_grid, dtype=float)
    return float(np.trapz(phi_vals * Q_vals, u_grid))


def risk_cdf_space(
    phi_grid: NDArray,
    Q_grid: NDArray,
    F_grid: NDArray,
) -> float:
    """Approximate R_phi(Q) via trapezoidal quadrature in CDF space.

    Uses the substitution u = F(x) so the integrand is phi(F(x)) * x.

    Parameters
    ----------
    phi_grid:
        phi(F(x)) evaluated at each x point, shape (n,).
    Q_grid:
        Loss values x on the support grid, shape (n,).
    F_grid:
        CDF values F(x) on the support grid, shape (n,).

    Returns
    -------
    float
        Numerical approximation of integral phi(F(x)) x dF(x).
    """
    phi_grid = np.asarray(phi_grid, dtype=float)
    Q_grid = np.asarray(Q_grid, dtype=float)
    F_grid = np.asarray(F_grid, dtype=float)
    # dF ~ differences of F_grid, midpoint rule
    integrand = phi_grid * Q_grid
    return float(np.trapz(integrand, F_grid))


# ---------------------------------------------------------------------------
# SpectralRiskMeasure class
# ---------------------------------------------------------------------------

class SpectralRiskMeasure:
    """A spectral risk measure R_phi(Q) = integral_0^1 phi(u) Q(u) du.

    Parameters
    ----------
    phi_callable:
        Spectrum function with signature phi(u, **phi_kwargs).
    **phi_kwargs:
        Fixed keyword arguments passed to *phi_callable* at every call.

    Examples
    --------
    >>> from src.spectral_risk import SpectralRiskMeasure, phi_power
    >>> srm = SpectralRiskMeasure(phi_power, beta=3.0)
    >>> import numpy as np
    >>> u = np.linspace(0, 1, 500)
    >>> risk = srm.evaluate_on_grid(np.sqrt(u), u)
    """

    def __init__(self, phi_callable: Callable[..., float | NDArray], **phi_kwargs) -> None:
        self.phi_callable = phi_callable
        self.phi_kwargs = phi_kwargs

    # ------------------------------------------------------------------
    def phi(self, u: float | NDArray) -> float | NDArray:
        """Evaluate the spectrum at *u*.

        Parameters
        ----------
        u:
            Probability level(s) in [0, 1].

        Returns
        -------
        float | NDArray
        """
        return self.phi_callable(u, **self.phi_kwargs)

    # ------------------------------------------------------------------
    def Phi(self, t: float) -> float:
        """Cumulative spectrum Phi(t) = integral_0^t phi(v) dv.

        Parameters
        ----------
        t:
            Upper limit in [0, 1].

        Returns
        -------
        float
        """
        return Phi_func(t, self.phi_callable, **self.phi_kwargs)

    # ------------------------------------------------------------------
    def evaluate_on_grid(
        self,
        Q_vals: NDArray,
        u_grid: NDArray,
    ) -> float:
        """Compute R_phi by trapezoidal rule on a pre-built grid.

        Parameters
        ----------
        Q_vals:
            Quantile values on *u_grid*, shape (n,).
        u_grid:
            Probability grid in [0, 1], shape (n,).

        Returns
        -------
        float
        """
        phi_vals = self.phi(np.asarray(u_grid, dtype=float))
        return risk_quantile_space(phi_vals, Q_vals, u_grid)

    # ------------------------------------------------------------------
    def evaluate_exact(self, Q_callable: Callable[[float], float]) -> float:
        """Compute R_phi by adaptive quadrature.

        Parameters
        ----------
        Q_callable:
            Quantile function Q : [0,1] -> R.

        Returns
        -------
        float
        """
        result, _ = quad(
            lambda u: float(self.phi(u)) * Q_callable(u),
            0.0, 1.0,
        )
        return float(result)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"SpectralRiskMeasure(phi={self.phi_callable.__name__}, kwargs={self.phi_kwargs})"
