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
        """Expected (noise, signal) cumulative proportions from current params."""

    def roc(self) -> tuple[np.ndarray, np.ndarray]:
        return self.compute_expected(smooth=True)


class ContinuousModel(Model):
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
    """High-Threshold detection model."""

    _param_spec = {"R": Param(initial=0.99)}

    def __init__(self, data: ROCData) -> None:
        super().__init__(data)
        self.R = 0.99

    def compute_expected(self, smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        model_noise = np.array([0, 1]) if smooth else self.data.noise_proportions
        model_signal = (1 - self.R) * model_noise + self.R
        return model_noise, model_signal


class SignalDetection(ContinuousModel):
    """Equal-variance signal detection model."""

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
    """Unequal-variance signal detection model."""

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
    """Dual Process (signal detection + threshold) model."""

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
