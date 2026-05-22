from collections.abc import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from kriterion.models import Model


def fit(
    model: Model,
    objective: Callable[[np.ndarray, Model], float],
    method: str = "L-BFGS-B",
) -> OptimizeResult:

    result = minimize(
        fun=objective,
        x0=model.x0,
        args=(model,),
        bounds=model.bounds,
        method=method,
        tol=1e-4,
    )
    model.update(result.x)
    return result
