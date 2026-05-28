from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from kriterion import objectives
from kriterion.models import Model

type ObjectiveFunction = Callable[[np.ndarray, np.ndarray, Model], float]


@dataclass
class ModelSummary:
    """Goodness-of-fit statistics for a fitted model.

    Attributes
    ----------
    dof :
        Degrees of freedom.
    chi2 :
        Pearson $\\chi^2$ statistic.
    chi2_p :
        $p$-value for the $\\chi^2$ statistic.
    g2 :
        Likelihood-ratio $G^2$ statistic.
    g2_p :
        $p$-value for the $G^2$ statistic.
    log_likelihood :
        Log-likelihood of the fitted model.
    aic :
        Akaike Information Criterion.
    bic :
        Bayesian Information Criterion.
    sse :
        Sum of squared errors between observed and expected cumulative proportions.
    """

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
    objective: ObjectiveFunction = objectives.log_likelihood_objective,
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

    Parameters
    ----------
    k :
        Number of estimated parameters in the model.
    ll :
        The log of the maximised value of the likelihood function for the model.
    """
    return float(2 * k - 2 * ll)


def bic(k: int, n: int, ll: float) -> float:
    """Bayesian Information Criterion

    $$
    k\\ln(n) - 2\\ln(\\hat{L})
    $$

    This statistic is useful for model comparisons.

    Parameters
    ----------
    k :
        Number of estimated parameters in the model.
    n :
        Total number of observations in the data.
    ll :
        The log of the maximised value of the likelihood function for the model.
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


def compare_nested(
    restricted: ModelSummary, full: ModelSummary
) -> tuple[float, int | float, np.ndarray]:
    """Likelihood-ratio test between two nested models.

    Tests whether the additional parameters of the fuller model yield a
    significant improvement in fit, using the difference in $G^2$ against a
    $\\chi^2$ distribution with degrees of freedom equal to the difference in
    parameter counts.

    Assumes the two models are nested: the restricted model must be obtainable
    by fixing one or more of the fuller model's parameters to constants. If
    they are not nested, the likelihood-ratio test is invalid and AIC or BIC
    should be used instead via $\\text{AIC}_a - \\text{AIC}_b$.

    Parameters
    ----------
    restricted :
        Fit summary of the simpler (restricted) model. This model should have fewer free
        parameters, and therefore larger residual degrees of freedom.
    full :
        Fit summary of the fuller model. This model should have more free parameters,
        and therefore smaller residual degrees of freedom.

    Returns
    -------
    tuple[float, int | float, np.ndarray]
        `(delta_g, delta_dof, p)`: the likelihood-ratio statistic
        $\\Delta G^2 = G^2_{\\text{restricted}} - G^2_{\\text{full}}$, the
        degrees of freedom $\\Delta\\text{dof}$, and the $p$-value.

    Raises
    ------
    ValueError
        If `full` does not have more parameters than `restricted` (i.e.
        `delta_dof <= 0`), or if the restricted model fits better than the
        fuller one (`delta_g < 0`), which shouldn't occur for correctly
        nested, correctly fitted models.
    """
    delta_g = restricted.g2 - full.g2
    delta_dof = restricted.dof - full.dof

    if delta_dof <= 0:
        raise ValueError(
            "`full` must have more parameters (smaller dof) than `restricted`"
        )

    if delta_g < 0:
        raise ValueError("restricted model fits better than full - check nesting/fit")

    p = stats.chi2.sf(delta_g, delta_dof)
    return delta_g, delta_dof, p
