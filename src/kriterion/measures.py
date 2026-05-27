from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

type ArrayLike = float | np.ndarray
"""A single float or a NumPy array."""


def _array_or_scalar(result: ArrayLike) -> ArrayLike:
    return float(result) if np.ndim(result) == 0 else result


@dataclass
class Performance:
    """Sensitivity and bias measures for one or more TPR/FPR pairs.

    Attributes
    ----------
    tpr :
        True positive rate(s).
    fpr :
        False positive rate(s).
    d_prime :
        Sensitivity $d'$.
    a_prime :
        Non-parametric sensitivity $A'$.
    c_bias :
        Criterion bias $c$.
    beta :
        Likelihood-ratio bias $\\beta$.
    a_z :
        Area under the binormal ROC curve $A_z$. Only present when
        ``z_intercept`` and ``z_slope`` are provided to
        :func:`compute_performance`.
    """

    tpr: ArrayLike
    fpr: ArrayLike
    d_prime: ArrayLike
    a_prime: ArrayLike
    c_bias: ArrayLike
    beta: ArrayLike
    a_z: ArrayLike | None = None


def compute_performance(
    tpr: ArrayLike,
    fpr: ArrayLike,
    z_intercept: ArrayLike | None = None,
    z_slope: ArrayLike | None = None,
) -> Performance:
    """Compute all sensitivity and bias measures for one or more TPR/FPR pairs.

    Parameters
    ----------
    tpr :
        True positive rate(s).
    fpr :
        False positive rate(s).
    z_intercept :
        Intercept of the fitted $z$-ROC line, required to compute $A_z$.
    z_slope :
        Slope of the fitted $z$-ROC line, required to compute $A_z$.
    """
    return Performance(
        tpr=tpr,
        fpr=fpr,
        d_prime=d_prime(tpr, fpr),
        a_prime=a_prime(tpr, fpr),
        c_bias=c_bias(tpr, fpr),
        beta=beta(tpr, fpr),
        a_z=(
            a_z(z_intercept, z_slope)
            if z_intercept is not None and z_slope is not None
            else None
        ),
    )


def d_prime(tpr: ArrayLike, fpr: ArrayLike) -> ArrayLike:
    """Sensitivity measure $d'$.

    $$
    d' = z(H) - z(F)
    $$

    Parameters
    ----------
    tpr : ArrayLike
        True positive rate(s), $0 \\leq H \\leq 1$.
    fpr : ArrayLike
        False positive rate(s), $0 \\leq F \\leq 1$.

    Returns
    -------
    ArrayLike
        $d'$. A float or array of the same shape as `tpr` and `fpr`.

    Notes
    -----
    Estimates the ability to discriminate between signal and noise. Larger
    magnitude indicates greater sensitivity.

    Examples
    --------
    >>> d = d_prime(0.75, 0.21)
    >>> print(d)
    1.480910997214322
    """
    return _array_or_scalar(norm.ppf(tpr) - norm.ppf(fpr))


def c_bias(tpr: ArrayLike, fpr: ArrayLike) -> ArrayLike:
    """Bias measure $c$.

    $$
    c = -\\frac{1}{2} \\left[ z(H) + z(F) \\right]
    $$


    Parameters
    ----------
    tpr : ArrayLike
        True positive rate(s), $0 \\leq H \\leq 1$.
    fpr : ArrayLike
        False positive rate(s), $0 \\leq F \\leq 1$.

    Returns
    -------
    ArrayLike
        $c$. A float or array of the same shape as `tpr` and `fpr`.

    Notes
    -----
    Estimates the criterion relative to the intersection of the signal and
    noise distributions. $c = 0$ indicates no bias; positive values indicate
    conservative bias, negative values indicate liberal bias.

    Examples
    --------
    >>> c = c_bias(0.75, 0.21)
    >>> print(c)
    0.06596574841107933
    """
    return _array_or_scalar(1 / 2 * -(norm.ppf(tpr) + norm.ppf(fpr)))


def a_prime(tpr: ArrayLike, fpr: ArrayLike) -> ArrayLike:
    """Non-parametric sensitivity index $A'$.

    $$
    A' = \\frac{1}{2} + \\operatorname{sgn}(H - F)
    \\frac{(H - F)^2 + |H - F|}
    {4 \\max(H, F) - 4  H  F}
    $$

    Parameters
    ----------
    tpr : ArrayLike
        True positive rate(s), $0 \\leq H \\leq 1$.
    fpr : ArrayLike
        False positive rate(s), $0 \\leq F \\leq 1$.

    Returns
    -------
    ArrayLike
        $A'$. A float or array of the same shape as `tpr` and `fpr`.

    Notes
    -----
    A non-parametric measure of discrimination that estimates sensitivity as
    the area under an "average" ROC curve for a single TPR/FPR pair. See
    Snodgrass & Corwin (1988). $d'$ is preferred where model assumptions hold.
    """
    return _array_or_scalar(
        0.5
        + np.sign(tpr - fpr)
        * ((tpr - fpr) ** 2 + np.abs(tpr - fpr))
        / (4 * np.maximum(tpr, fpr) - 4 * tpr * fpr)
    )


def beta(tpr: ArrayLike, fpr: ArrayLike) -> ArrayLike:
    """Likelihood-ratio bias measure $\\beta$.

    $$
    \\beta = \\exp\\!\\left(\\frac{z(F)^2 - z(H)^2}{2}\\right)
    $$

    Parameters
    ----------
    tpr : ArrayLike
        True positive rate(s), $0 \\leq H \\leq 1$.
    fpr : ArrayLike
        False positive rate(s), $0 \\leq F \\leq 1$.

    Returns
    -------
    ArrayLike
        $\\beta$. A float or array of the same shape as `tpr` and `fpr`.

    Notes
    -----
    The ratio of the signal distribution density to the noise distribution
    density at the criterion. $\\beta = 1$ indicates no bias; $\\beta > 1$
    indicates conservative bias, $\\beta < 1$ indicates liberal bias. See
    Snodgrass & Corwin (1988) for details.
    """
    return _array_or_scalar(np.exp((norm.ppf(fpr) ** 2 - norm.ppf(tpr) ** 2) / 2))


def beta_doubleprime(
    tpr: ArrayLike,
    fpr: ArrayLike,
    donaldson: bool = False,
) -> ArrayLike:
    """Non-parametric bias index $\\beta''$.

    Grier's (1971) formula:

    $$
    \\beta'' = \\text{sgn}(H - F) \\frac{H (1-H) - F (1-F)} {H (1-H) + F (1-F)}
    $$

    Parameters
    ----------
    tpr : ArrayLike
        True positive rate(s), $0 \\leq H \\leq 1$.
    fpr : ArrayLike
        False positive rate(s), $0 \\leq F \\leq 1$.
    donaldson : bool, optional
        Use Donaldson's formula instead of Grier's, by default False.

    Returns
    -------
    ArrayLike
        $\\beta''$. A float or array of the same shape as `tpr` and `fpr`.

    Notes
    -----
    $\\beta'' \\in [-1, 1]$. Positive values indicate conservative bias,
    negative values indicate liberal bias. Default is Grier's (1971) as cited
    in Stanislaw & Todorov (1999).
    """
    fnr, tnr = 1 - tpr, 1 - fpr

    if donaldson:
        return _array_or_scalar(
            ((fnr * tnr) - (tpr * fpr)) / ((fnr * tnr) + (tpr * fpr))
        )

    return _array_or_scalar(
        np.sign(tpr - fpr) * (tpr * fnr - fpr * tnr) / (tpr * fnr + fpr * tnr)
    )


def a_z(z_intercept: ArrayLike, z_slope: ArrayLike) -> ArrayLike:
    """Area under the binormal ROC curve, $A_z$.

    $$
    A_z = \\Phi \\left( \\frac{a}{\\sqrt{1 + b^2}} \\right)
    $$

    where $a$ is the $z$-ROC intercept, $b$ is the $z$-ROC slope, and
    $\\Phi$ is the standard normal CDF.

    Parameters
    ----------
    z_intercept : ArrayLike
        Intercept $a$ of the fitted $z$-ROC line.
    z_slope : ArrayLike
        Slope $b$ of the fitted $z$-ROC line.

    Returns
    -------
    ArrayLike
        $A_z$, the area under the binormal ROC curve.

    Notes
    -----
    Useful for checking the equal-variance $d'$ assumption: the slope $b$
    should equal 1 (equivalently, $\\ln b = 0$) under equal variances.
    See Stanislaw & Todorov (1999).
    """
    return _array_or_scalar(norm.cdf(z_intercept / np.sqrt(1 + z_slope**2)))


if __name__ == "__main__":
    print(d_prime(0.75, 0.21))
    print(c_bias(0.75, 0.21))
    print(a_prime(0.75, 0.21))
    print(beta(0.75, 0.21))
    print(beta_doubleprime(0.75, 0.21))
    print(beta_doubleprime(0.75, 0.21, donaldson=True))
    print(compute_performance(0.75, 0.21, 1.23, 0.79))
