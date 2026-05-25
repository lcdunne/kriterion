from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from kriterion import objectives
from kriterion.models import Model


@dataclass
class ModelSummary:
    """Model Summary with all statistics calculated."""

    dof: int | float
    chi2: float
    chi2_p: float
    g2: float
    g2_p: float
    log_likelihood: float
    aic: float
    bic: float
    sse: float


def fit(
    model: Model,
    objective: Callable[[np.ndarray, np.ndarray, Model], float],
    method: str = "L-BFGS-B",
) -> ModelSummary:
    """Fit a theoretical model to observed data.

    Parameters
    ----------
    model :
        An instance of a model subclass, e.g. an instance of `SignalDetection`.
    objective :
        One of the objective functions, e.g. $G^2$.
    method:
        The type of solver to use (see `scipy.optimize.minimize`). Note that some are
        incompatible for fitting detection models.
    """

    # This closure wraps common procedure on each opt iteration.
    def _obj(x: np.ndarray) -> float:
        model.update(x)
        noise_exp, signal_exp = model.compute_expected()
        return objective(signal_exp, noise_exp, model)

    result = minimize(
        fun=_obj, x0=model.x0, bounds=model.bounds, method=method, tol=1e-8
    )

    if not result.success:
        raise Exception(
            f"Failed to fit {model.__class__.__name__} using {objective.__name__}"
        )

    model.update(result.x)

    return _calculate_all_stats(model)


def aic(k: int, ll: float) -> float:
    """Akaike's Information Criterion:

    $$
    2k-2\\ln(\\hat{L})
    $$

    This statistic is useful for model comparisons.
    """
    return float(2 * k - 2 * ll)


def bic(k: int, n: int, ll: float) -> float:
    """Bayesian Information Criterion

    $$
    k\\ln(n) - 2\\ln(\\hat{L})
    $$

    This statistic is useful for model comparisons.
    """
    return float(k * np.log(n) - 2 * ll)


def _calculate_all_stats(model: Model) -> ModelSummary:
    noise_exp, signal_exp = model.compute_expected()
    chi2 = objectives.chi_squared_objective(signal_exp, noise_exp, model)
    chi2_p = float(1 - stats.chi2.cdf(chi2, model.dof))
    g2 = objectives.g_squared_objective(signal_exp, noise_exp, model)
    g2_p = float(1 - stats.chi2.cdf(g2, model.dof))  # G^2 is chi^2 distributed
    ll = -objectives.log_likelihood_objective(signal_exp, noise_exp, model)
    _aic = aic(model.n_params, ll)
    _bic = bic(model.n_params, model.data.n_signal + model.data.n_noise, ll)

    sse = objectives.sse_objective(signal_exp, noise_exp, model)
    return ModelSummary(
        dof=model.dof,
        chi2=chi2,
        chi2_p=chi2_p,
        g2=g2,
        g2_p=g2_p,
        log_likelihood=ll,
        aic=_aic,
        bic=_bic,
        sse=sse,
    )
