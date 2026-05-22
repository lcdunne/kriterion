import numpy as np

from kriterion.models import Model


def g_squared(obs_prop: np.ndarray, exp_prop: np.ndarray, n: int) -> float:
    """Binomial $G^2$ (likelihood-ratio) statistic for a single response class.

    Parameters
    ----------
    obs_prop :
        Observed cumulative proportions (n_criteria values).
    exp_prop :
        Expected cumulative proportions from the model.
    n :
        Total count for this response class.
    """
    obs_count = obs_prop * n
    above = obs_count * np.log(obs_prop / exp_prop)  # Above criterion
    below = (n - obs_count) * np.log((1 - obs_prop) / (1 - exp_prop))  # Below criterion
    return 2 * float(np.sum(above + below))


def g_squared_objective(model: Model, x: np.ndarray) -> float:
    model.update(x)
    noise_exp, signal_exp = model.compute_expected()
    return g_squared(
        model.data.signal_proportions, signal_exp, model.data.n_signal
    ) + g_squared(model.data.noise_proportions, noise_exp, model.data.n_noise)
