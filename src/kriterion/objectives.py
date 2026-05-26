import numpy as np
from scipy.special import xlogy

from kriterion.models import Model


def chi_squared(obs: np.ndarray, exp: np.ndarray) -> float:
    """The $\\chi^2$ statistic for a single response class.

    $$
    \\chi^2 = \\sum_{k=1}^{K} \\frac{(O_k - E_k)^2}{E_k}
    $$

    where $O_k$ and $E_k$ are the observed and expected cell counts in bin $k$.

    Parameters
    ----------
    obs :
        Observed cell counts $O_k$ per rating bin.
    exp :
        Expected cell counts $E_k$ from the model.
    """
    return float(np.sum((obs - exp) ** 2 / exp))


def chi_squared_cumulative(obs_prop: np.ndarray, exp_prop: np.ndarray, n: int) -> float:
    """The $\\chi^2$ over binary (above/below) splits at each criterion threshold.

    This implementation is provided as an alternative approach for optimisation that may
    result in a better fit; however it is not recommended for statistical analysis/model
    comparisons due to violation of the assumption of independence.

    Each of the $K-1$ thresholds divides the $n$ trials into two cells: those
    rated at or above threshold $k$, and those below. The $\\chi^2$
    statistic is summed over both cells of every threshold:

    $$
    \\chi^2 = \\sum_{k=1}^{K-1} \\sum_{j \\in \\{a,\\,b\\}}
        \\frac{(O_{kj} - E_{kj})^2}{E_{kj}}
    $$

    where at threshold $k$ the above-cell is $(O_{ka}, E_{ka}) = (O_k, E_k)$ and
    the below-cell is $(O_{kb}, E_{kb}) = (N - O_k,\\, N - E_k)$. Expanding the
    inner sum over both cells:

    $$
    \\chi^2 = \\sum_{k=1}^{K-1} \\left[
        \\frac{(O_k - E_k)^2}{E_k}
        + \\frac{((N - O_k) - (N - E_k))^2}{N - E_k}
    \\right]
    $$

    where $O_k$ and $E_k$ are the observed and expected cumulative counts at
    threshold $k$, and $N$ is the total trial count.

    Parameters
    ----------
    obs_prop :
        Observed cumulative proportions $\\hat{p}_k$ at each threshold.
    exp_prop :
        Expected cumulative proportions $p_k$ from the model.
    n :
        Total trial count for this response class.
    """
    obs_count = obs_prop * n
    exp_count = exp_prop * n
    delta_sq = (obs_count - exp_count) ** 2
    above = delta_sq / exp_count
    below = delta_sq / (n - exp_count)
    return float(np.sum(above + below))


def g_squared(obs: np.ndarray, exp: np.ndarray) -> float:
    """The $G^2$ (likelihood-ratio) statistic for a single response class[^cressie_read_1984]:

    $$
    G^2 = 2 \\sum_{k=1}^{K} O_k \\ln \\left( \\frac{O_k}{E_k} \\right)
    $$

    where $O_k$ and $E_k$ are the observed and expected cell counts in bin $k$.

    [^cressie_read_1984]: [Cressie, N., & Timothy R. C. Read. (1984). Multinomial
    Goodness-of-Fit Tests. Journal of the Royal Statistical Society. Series B
    (Methodological), 46(3), 440–464](http://www.jstor.org/stable/2345686).

    Parameters
    ----------
    obs :
        Observed cell counts $O_k$ per rating bin.
    exp :
        Expected cell counts $E_k$ from the model.
    """
    return 2 * float(np.sum(xlogy(obs, obs / exp)))


def g_squared_cumulative(obs_prop: np.ndarray, exp_prop: np.ndarray, n: int) -> float:
    """The $G^2$ (likelihood-ratio) statistic over binary (above/below) splits at each criterion threshold.

    This implementation is provided as an alternative approach for optimisation that may
    result in a better fit; however it is not recommended for statistical analysis/model
    comparisons due to violation of the assumption of independence.

    Each of the $K-1$ thresholds divides the $n$ trials into two cells: those
    rated at or above threshold $k$, and those below. The $G^2$
    statistic is summed over both cells of every threshold:

    $$
    G^2 = 2n \\sum_{k=1}^{K-1} \\sum_{j \\in \\{a,\\,b\\}}
        \\hat{p}_{kj} \\ln \\left( \\frac{\\hat{p}_{kj}}{p_{kj}} \\right)
    $$

    where at threshold $k$ the above-cell is $(\\hat{p}_{ka}, p_{ka}) = (\\hat{p}_k, p_k)$ and
    the below-cell is $(\\hat{p}_{kb}, p_{kb}) = (1 - \\hat{p}_k,\\, 1 - p_k)$. Expanding the
    inner sum over both cells:

    $$
    G^2 = 2n \\sum_{k=1}^{K-1} \\left[
        \\hat{p}_k \\ln \\left( \\frac{\\hat{p}_k}{p_k} \\right) + (1 - \\hat{p}_k) \\ln \\left( \\frac{1 - \\hat{p}_k}{1 - p_k} \\right)
    \\right]
    $$

    where $\\hat{p}_k$ and $p_k$ are the observed and expected cumulative proportions
    at threshold $k$, and $n$ is the total trial count.

    Parameters
    ----------
    obs_prop :
        Observed cumulative proportions $\\hat{p}_k$ at each threshold.
    exp_prop :
        Expected cumulative proportions $p_k$ from the model.
    n :
        Total trial count for this response class.
    """
    above = xlogy(obs_prop, obs_prop / exp_prop)
    below = xlogy((1 - obs_prop), (1 - obs_prop) / (1 - exp_prop))
    return 2 * n * float(np.sum(above + below))


def log_likelihood(obs: np.ndarray, exp_prop: np.ndarray) -> float:
    """Log-likelihood for a single response class.

    The log-likelihood of the observed cell counts given predicted
    cell probabilities is:

    $$
    \\ell = \\sum_{k=1}^{K} O_k \\ln p_k
    $$

    where $O_k$ are observed cell counts and $p_k$ are predicted cell
    probabilities. Returns a negative value.

    Parameters
    ----------
    obs :
        Observed cell counts $O_k$ per rating bin.
    exp_prop :
        Predicted cell probabilities $p_k$ from the model.
    """
    return float(np.sum(xlogy(obs, exp_prop)))


def sse(obs_prop: np.ndarray, exp_prop: np.ndarray) -> float:
    """Sum of squared errors for a single response class.

    Calculates the $\\text{SSE}$ between the observevd and expected cumulative proportions.

    Parameters
    ----------
    obs_prop :
        Observed cumulative proportions at each threshold.
    exp_prop :
        Expected cumulative proportions from the model.
    """
    return float(np.sum((obs_prop - exp_prop) ** 2))


def _to_cell_counts(cumulative_props: np.ndarray, n: int) -> np.ndarray:
    return np.maximum(np.diff(cumulative_props, prepend=0.0, append=1.0), 1e-10) * n


def chi_squared_objective(
    signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model
) -> float:
    return chi_squared(
        model.data.signal.astype(float),
        _to_cell_counts(signal_exp, model.data.n_signal),
    ) + chi_squared(
        model.data.noise.astype(float), _to_cell_counts(noise_exp, model.data.n_noise)
    )


def chi_squared_cumulative_objective(
    signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model
) -> float:
    return chi_squared_cumulative(
        model.data.signal_proportions, signal_exp, model.data.n_signal
    ) + chi_squared_cumulative(
        model.data.noise_proportions, noise_exp, model.data.n_noise
    )


def g_squared_objective(
    signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model
) -> float:
    return g_squared(
        model.data.signal.astype(float),
        _to_cell_counts(signal_exp, model.data.n_signal),
    ) + g_squared(
        model.data.noise.astype(float), _to_cell_counts(noise_exp, model.data.n_noise)
    )


def g_squared_cumulative_objective(
    signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model
) -> float:
    return g_squared_cumulative(
        model.data.signal_proportions, signal_exp, model.data.n_signal
    ) + g_squared_cumulative(
        model.data.noise_proportions, noise_exp, model.data.n_noise
    )


def log_likelihood_objective(
    signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model
) -> float:
    signal_p = np.maximum(np.diff(signal_exp, prepend=0.0, append=1.0), 1e-10)
    noise_p = np.maximum(np.diff(noise_exp, prepend=0.0, append=1.0), 1e-10)
    return -(
        log_likelihood(model.data.signal.astype(float), signal_p)
        + log_likelihood(model.data.noise.astype(float), noise_p)
    )


def sse_objective(signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model) -> float:
    return sse(model.data.signal_proportions, signal_exp) + sse(
        model.data.noise_proportions, noise_exp
    )
