from collections.abc import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from kriterion.models import Model


def fit(
    model: Model,
    objective: Callable[[np.ndarray, np.ndarray, Model], float],
    method: str = "L-BFGS-B",
) -> OptimizeResult:

    # This closure wraps common procedure on each opt iteration.
    def _obj(x: np.ndarray) -> float:
        model.update(x)
        noise_exp, signal_exp = model.compute_expected()
        return objective(signal_exp, noise_exp, model)

    result = minimize(
        fun=_obj, x0=model.x0, bounds=model.bounds, method=method, tol=1e-4
    )

    model.update(result.x)
    return result
