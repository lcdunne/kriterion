from enum import StrEnum
from functools import cached_property

import numpy as np
from scipy.stats import norm


class Correction(StrEnum):
    """Method for correcting extreme proportions (0s and 1s) in ROC data.

    Extreme proportions produce infinite $z$-scores and must be adjusted
    before converting to $z$-space. See Stanislaw & Todorov (1999) for reference
    on these methods. The default value is `INCREMENTAL`.

    Attributes
    ----------
    NONE :
        No correction applied.
    INCREMENTAL :
        Adds i/k to each cumulative frequency and 1 to the total, where
        i is the bin index and k is the number of bins. Default.
    LOGLINEAR :
        Adds 0.5 to all frequencies and 1 to the total (Hautus, 1995).
    EXTREME :
        Corrects only 0s and 1s: 0 is corrected to 0.5/n, 1 is corrected to (n - 0.5)/n
        (Macmillan & Kaplan, 1985).
    """

    NONE = "none"
    INCREMENTAL = "incremental"
    LOGLINEAR = "loglinear"
    EXTREME = "extreme"


def _proportions_uncorrected(arr: np.ndarray) -> np.ndarray:
    """Cumulative proportions with no correction applied."""
    return (np.cumsum(arr) / arr.sum())[:-1]


def _proportions_incremental(arr: np.ndarray) -> np.ndarray:
    cumfreqs = np.cumsum(arr)
    k = len(arr)
    n = int(cumfreqs[-1])
    i = np.arange(1, k + 1)
    return ((cumfreqs + i / k) / (n + 1))[:-1]


def _proportions_loglinear(arr: np.ndarray) -> np.ndarray:
    n = arr.sum()
    return ((np.cumsum(arr) + 0.5) / (n + 1))[:-1]


def _proportions_extreme(arr: np.ndarray) -> np.ndarray:
    n = arr.sum()
    props = np.cumsum(arr) / n
    props = np.where(props == 0, 0.5 / n, props)
    props = np.where(props == 1, (n - 0.5) / n, props)
    return props[:-1]


def compute_proportions(
    arr: np.ndarray,
    correction: Correction = Correction.INCREMENTAL,
) -> np.ndarray:
    """Compute cumulative proportions from frequency counts.

    Parameters
    ----------
    arr : np.ndarray
        Frequency counts per rating bin, ordered from strongest signal
        to strongest noise.
    correction : Correction, optional
        Correction method to apply to avoid 0s and 1s in the output,
        by default `Correction.INCREMENTAL`.

    Returns
    -------
    np.ndarray
        Cumulative proportions with the final 1.0 omitted (n - 1 values).
    """
    match correction:
        case Correction.NONE:
            return _proportions_uncorrected(arr)
        case Correction.INCREMENTAL:
            return _proportions_incremental(arr)
        case Correction.LOGLINEAR:
            return _proportions_loglinear(arr)
        case Correction.EXTREME:
            return _proportions_extreme(arr)


class ROCData:
    """Observed frequency counts for a rating-scale ROC experiment.

    Parameters
    ----------
    signal : list[int]
        Frequency counts per rating bin for signal trials.
    noise : list[int]
        Frequency counts per rating bin for noise trials.
    condition : str, optional
        Label for the experimental condition, by default None.
    correction : Correction, optional
        Correction method for extreme proportions (0s and 1s),
        by default `Correction.INCREMENTAL`.

    Raises
    ------
    ValueError
        If `signal` and `noise` have different numbers of bins.
    """

    def __init__(
        self,
        signal: list[int],
        noise: list[int],
        condition: str | None = None,
        correction: Correction = Correction.INCREMENTAL,
    ) -> None:
        if len(signal) != len(noise):
            raise ValueError("signal and noise must have the same number of bins")
        self.signal = np.asarray(signal)
        self.noise = np.asarray(noise)
        self.condition = condition
        self.correction = correction

    @cached_property
    def n_signal(self) -> int:
        return int(self.signal.sum())

    @cached_property
    def n_noise(self) -> int:
        return int(self.noise.sum())

    @cached_property
    def signal_proportions(self) -> np.ndarray:
        return compute_proportions(self.signal, self.correction)

    @cached_property
    def noise_proportions(self) -> np.ndarray:
        return compute_proportions(self.noise, self.correction)

    @cached_property
    def z_signal(self) -> np.ndarray:
        return norm.ppf(self.signal_proportions)

    @cached_property
    def z_noise(self) -> np.ndarray:
        return norm.ppf(self.noise_proportions)
