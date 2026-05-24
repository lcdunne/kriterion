from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy import stats

from kriterion.data import ROCData


@dataclass
class Param:
    initial: float
    bounds: tuple[float | None, float | None] = (None, None)


class Model(ABC):
    _param_spec: ClassVar[dict[str, Param]]

    def __init__(self, data: ROCData) -> None:
        self.data = data

    @property
    def n_params(self) -> int:
        return len(self._param_spec)

    @property
    def dof(self) -> int:
        return 2 * (len(self.data.signal) - 1) - self.n_params

    @property
    def parameters(self) -> dict[str, float]:
        return {p: float(getattr(self, p)) for p in self._param_spec}

    @property
    def x0(self) -> np.ndarray:
        return np.array([getattr(self, p) for p in self._param_spec])

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        return [p.bounds for p in self._param_spec.values()]

    def update(self, x: np.ndarray) -> None:
        for i, p in enumerate(self._param_spec.keys()):
            self.__setattr__(p, x[i])

    @abstractmethod
    def compute_expected(self, smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Expected cumulative proportions under the current parameter values.

        All subclasses must implement this method.

        Parameters
        ----------
        smooth : bool, optional
            If True, evaluate over a dense criterion grid for plotting rather
            than at the observed rating boundaries, by default False.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            `(noise, signal)` cumulative proportions.
        """

    def roc(self) -> tuple[np.ndarray, np.ndarray]:
        """Smooth ROC curve for plotting.

        Evaluates the model over a dense criterion grid rather than the
        observed rating boundaries.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            `(noise, signal)` cumulative proportions over the dense grid.
        """
        return self.compute_expected(smooth=True)


class ContinuousModel(Model):
    """Base class for models with continuous criterion parameters.

    Extends `Model` with a set of free criterion locations $c_k$.

    Attributes
    ----------
    criteria : np.ndarray
        Criterion locations $c_k$, one per rating boundary, initialised to
        a uniform grid over $[-1.5, 1.5]$.
    """

    criteria: np.ndarray

    def __init__(self, data: ROCData) -> None:
        super().__init__(data)
        self._init_criteria()

    @property
    def n_params(self) -> int:
        return super().n_params + len(self.criteria)

    @property
    def parameters(self) -> dict[str, float]:
        return {
            **super().parameters,
            **{f"c{i}": float(c) for i, c in enumerate(self.criteria)},
        }

    @property
    def x0(self) -> np.ndarray:
        return np.concatenate([super().x0, self.criteria])

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        return super().bounds + [(None, None)] * len(self.criteria)

    def _init_criteria(self) -> None:
        self.criteria = np.linspace(1.5, -1.5, len(self.data.signal) - 1)

    def update(self, x: np.ndarray) -> None:
        super().update(x)
        n_named = len(self._param_spec)
        self.criteria = x[n_named:]


class HighThreshold(Model):
    """High-threshold detection model.

    Signal responses are modelled as a mix of true detection with probability $R$ and
    guessing with probability $(1-R)G$:

    $$
    H_k = R + (1 - R) \\cdot F_k
    $$

    Attributes
    ----------
    R : float
        Detection probability, $0 \\leq R \\leq 1$.
    """

    _param_spec = {"R": Param(initial=0.99)}

    def __init__(self, data: ROCData) -> None:
        super().__init__(data)
        self.R = 0.99

    def compute_expected(self, smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        model_noise = np.array([0, 1]) if smooth else self.data.noise_proportions
        model_signal = (1 - self.R) * model_noise + self.R
        return model_noise, model_signal


class SignalDetection(ContinuousModel):
    """Equal-variance signal detection model[^macmillan_creelman].

    Signal and noise are modelled as Gaussian distributions with equal variance,
    separated by sensitivity $d'$:

    $$
    H_k = \\Phi\\left(\\frac{d'}{2} - c_k\\right), \\quad
    F_k = \\Phi\\left(-\\frac{d'}{2} - c_k\\right)
    $$

    where $\\Phi$ is the standard normal CDF and $c_k$ are the criterion
    locations.

    [^macmillan_creelman]: [Macmillan, N.A., & Creelman, C.D. (2004). Detection Theory:
    A User's Guide (2nd ed.). Psychology Press.](https://doi.org/10.4324/9781410611147)

    Attributes
    ----------
    d : float
        Sensitivity $d'$.
    criteria : np.ndarray
        Criterion locations $c_k$, one per rating boundary.
    """

    _param_spec = {"d": Param(initial=1.0)}

    d: float

    def __init__(self, data: ROCData) -> None:
        super().__init__(data)
        self.d = 1.0

    def compute_expected(self, smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        c = np.linspace(3.0, -3.0, 200) if smooth else self.criteria
        model_signal = stats.norm.cdf(self.d / 2 - c)
        model_noise = stats.norm.cdf(-self.d / 2 - c)
        return model_noise, model_signal


class UnequalSignalDetection(ContinuousModel):
    """Unequal-variance signal detection model.

    Extends the equal-variance model by allowing the signal distribution to
    have standard deviation $\\sigma_s \\neq 1$:

    $$
    H_k = \\Phi\\left(\\frac{d'/2 - c_k}{\\sigma_s}\\right), \\quad
    F_k = \\Phi\\left(-\\frac{d'}{2} - c_k\\right)
    $$

    where $\\Phi$ is the standard normal CDF and $c_k$ are the criterion
    locations.

    Attributes
    ----------
    d : float
        Sensitivity $d'$.
    signal_sd : float
        Standard deviation $\\sigma_s$ of the signal distribution.
    criteria : np.ndarray
        Criterion locations $c_k$, one per rating boundary.
    """

    _param_spec = {
        "d": Param(initial=1.0),
        "signal_sd": Param(initial=1.5, bounds=(0, None)),
    }

    d: float
    signal_sd: float

    def __init__(self, data: ROCData) -> None:
        super().__init__(data)
        self.d = 1.0
        self.signal_sd = 1.5

    def compute_expected(self, smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        c = np.linspace(3.0, -3.0, 200) if smooth else self.criteria
        model_signal = stats.norm.cdf(self.d / 2 - c, scale=self.signal_sd)
        model_noise = stats.norm.cdf(-self.d / 2 - c)
        return model_noise, model_signal


class DualProcess(ContinuousModel):
    """Dual-process signal detection model[^yonelinas_et_al_1996].

    Combines continuous Gaussian discrimination with a high-threshold
    recollection component of probability $R$:

    $$
    H_k = R + (1 - R) \\cdot \\Phi\\left(\\frac{d'}{2} - c_k\\right), \\quad
    F_k = \\Phi\\left(-\\frac{d'}{2} - c_k\\right)
    $$

    where $\\Phi$ is the standard normal CDF and $c_k$ are the criterion
    locations .

    [^yonelinas_et_al_1996]: [Yonelinas, A. P., Dobbins, I., Szymanski, M. D.,
    Dhaliwal, H. S., & King, L. (1996). Signal-Detection, Threshold, and
    Dual-Process Models of Recognition Memory: ROCs and Conscious Recollection.
    Consciousness and Cognition, 5(4), 418–441.](https://doi.org/10.1006/CCOG.1996.0026)

    Attributes
    ----------
    d : float
        Continuous sensitivity $d'$.
    R : float
        Recollection probability, $0 \\leq R \\leq 1$.
    criteria : np.ndarray
        Criterion locations $c_k$, one per rating boundary.
    """

    _param_spec = {
        "d": Param(initial=1.0),
        "R": Param(initial=0.99, bounds=(0, 1)),
    }

    d: float
    R: float

    def __init__(self, data: ROCData) -> None:
        super().__init__(data)
        self.d = 1.0
        self.R = 0.9

    def compute_expected(self, smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        c = np.linspace(3.0, -3.0, 200) if smooth else self.criteria
        model_signal = self.R + (1 - self.R) * stats.norm.cdf(self.d / 2 - c)
        model_noise = stats.norm.cdf(-self.d / 2 - c)
        return model_noise, model_signal
