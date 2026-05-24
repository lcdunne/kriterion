import numpy as np
from scipy.special import xlogy

from kriterion.models import Model


def chi_squared(obs: np.ndarray, exp: np.ndarray) -> float:
    """Standard $\\chi^2$ statistic for a single response class.

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


def chi_squared_binom(obs_prop: np.ndarray, exp_prop: np.ndarray, n: int) -> float:
    """Binomial $\\chi^2$ statistic for a single response class.

    Each criterion threshold $k$ divides the $n$ trials into two groups: those
    rated at or above $k$, and those below. $\\chi^2$ sums the Pearson statistic
    across all $K-1$ thresholds:

    $$
    \\chi^2 = \\sum_{k=1}^{K-1} \\left[
        \\frac{(O_k - E_k)^2}{E_k} + \\frac{((N-O_k) - (N-E_k))^2}{N - E_k}
    \\right]
    $$

    where $O_k$ and $E_k$ are the observed and expected cumulative counts at
    threshold $k$, and $N$ is the total trial count.

    Parameters
    ----------
    obs_prop :
        Observed cumulative proportions $O_k / N$ at each threshold.
    exp_prop :
        Expected cumulative proportions $E_k / N$ from the model.
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
    """$G^2$ (likelihood-ratio) statistic for a single response class. Standard
    (multinomial) variant [^cressie_read_1984]:

    $$
    G^2 = 2 \\sum_{k=1}^{K} O_k \\ln \\frac{O_k}{E_k}
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


def g_squared_binom(obs_prop: np.ndarray, exp_prop: np.ndarray, n: int) -> float:
    """Binomial $G^2$ (likelihood-ratio) statistic for a single response class.

    Each criterion threshold $k$ divides the $n$ trials into two groups: those
    rated at or above $k$, and those below. $G^2$ sums the log-likelihood ratio
    across all $K-1$ thresholds:

    $$
    G^2 = 2n \\sum_{k=1}^{K-1} \\left[
        O_k \\ln \\frac{O_k}{E_k} + (1 - O_k) \\ln \\frac{1 - O_k}{1 - E_k}
    \\right]
    $$

    where $O_k$ and $E_k$ are the observed and expected cumulative proportions
    at threshold $k$, and $n$ is the total trial count. Terms where $O_k = 0$
    or $O_k = 1$ contribute zero by convention.

    Parameters
    ----------
    obs_prop :
        Observed cumulative proportions $O_k$ at each threshold.
    exp_prop :
        Expected cumulative proportions $E_k$ from the model.
    n :
        Total trial count for this response class.
    """
    above = xlogy(obs_prop, obs_prop / exp_prop)
    below = xlogy((1 - obs_prop), (1 - obs_prop) / (1 - exp_prop))
    return 2 * n * float(np.sum(above + below))


def log_likelihood(obs: np.ndarray, exp_prop: np.ndarray) -> float:
    """Log-likelihood for a single response class.

    The multinomial log-likelihood of the observed cell counts given predicted
    cell probabilities is:

    $$
    \\LL = \\sum_{k=1}^{K} O_k \\ln P_k
    $$

    where $O_k$ are observed cell counts and $P_k$ are predicted cell
    probabilities. Returns a negative value.

    Parameters
    ----------
    obs :
        Observed cell counts $O_k$ per rating bin.
    exp_prop :
        Predicted cell probabilities $P_k$ from the model.
    """
    return float(np.sum(xlogy(obs, exp_prop)))


def log_likelihood_binom(obs_prop: np.ndarray, exp_prop: np.ndarray, n: int) -> float:
    """Binomial pseudo-log-likelihood for a single response class.

    Each criterion threshold $k$ is treated as an independent binary split.
    The log-likelihood is summed across all $K-1$ thresholds:

    $$
    \\log L = n \\sum_{k=1}^{K-1} \\left[
        O_k \\ln E_k + (1 - O_k) \\ln(1 - E_k)
    \\right]
    $$

    where $O_k$ and $E_k$ are the observed and expected cumulative proportions
    at threshold $k$, and $n$ is the total trial count. Returns a negative value.

    Parameters
    ----------
    obs_prop :
        Observed cumulative proportions $O_k$ at each threshold.
    exp_prop :
        Expected cumulative proportions $E_k$ from the model.
    n :
        Total trial count for this response class.
    """
    return n * float(
        np.sum(xlogy(obs_prop, exp_prop) + xlogy(1 - obs_prop, 1 - exp_prop))
    )


def sse(obs_prop: np.ndarray, exp_prop: np.ndarray) -> float:
    """Sum of squared errors for a single response class.

    Calculates the $SSE$ between the observevd and expected cumulative proportions.

    Parameters
    ----------
    obs_prop :
        Observed cumulative proportions at each threshold.
    exp_prop :
        Expected cumulative proportions from the model.
    """
    return np.sum((obs_prop - exp_prop) ** 2)


def _to_cell_counts(cumulative_props: np.ndarray, n: int) -> np.ndarray:
    return np.maximum(np.diff(cumulative_props, prepend=0.0, append=1.0), 1e-10) * n


def log_likelihood_binom_objective(
    signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model
) -> float:
    return -(
        log_likelihood_binom(
            model.data.signal_proportions, signal_exp, model.data.n_signal
        )
        + log_likelihood_binom(
            model.data.noise_proportions, noise_exp, model.data.n_noise
        )
    )


def chi_squared_objective(
    signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model
) -> float:
    return chi_squared(
        model.data.signal.astype(float),
        _to_cell_counts(signal_exp, model.data.n_signal),
    ) + chi_squared(
        model.data.noise.astype(float), _to_cell_counts(noise_exp, model.data.n_noise)
    )


def chi_squared_binom_objective(
    signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model
) -> float:
    return chi_squared_binom(
        model.data.signal_proportions, signal_exp, model.data.n_signal
    ) + chi_squared_binom(model.data.noise_proportions, noise_exp, model.data.n_noise)


def g_squared_objective(
    signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model
) -> float:
    return g_squared(
        model.data.signal.astype(float),
        _to_cell_counts(signal_exp, model.data.n_signal),
    ) + g_squared(
        model.data.noise.astype(float), _to_cell_counts(noise_exp, model.data.n_noise)
    )


def g_squared_binom_objective(
    signal_exp: np.ndarray, noise_exp: np.ndarray, model: Model
) -> float:
    return g_squared_binom(
        model.data.signal_proportions, signal_exp, model.data.n_signal
    ) + g_squared_binom(model.data.noise_proportions, noise_exp, model.data.n_noise)


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
