import numpy as np
import pytest

from kriterion.measures import (
    Performance,
    a_prime,
    a_z,
    beta,
    beta_doubleprime,
    c_bias,
    compute_performance,
    d_prime,
)


@pytest.mark.parametrize(
    "tpr,fpr,expected",
    [
        (0.5, 0.5, 0.0),  # chance
        (0.20, 0.80, -1.683),  # below chance gives negative d'
        (0.95, 0.20, 2.486),
    ],
)
def test_d_prime(tpr, fpr, expected):
    assert d_prime(tpr, fpr) == pytest.approx(expected, abs=1e-3)


def test_d_prime_boundary():
    assert np.isposinf(d_prime(1.0, 0.0))
    assert np.isneginf(d_prime(0.0, 1.0))


@pytest.mark.parametrize(
    "tpr,fpr,expected",
    [
        (0.5, 0.5, 0.0),  # unbiased
        (0.95, 0.20, -0.402),  # liberal
        (0.90, 0.05, 0.182),  # conservative
    ],
)
def test_c_bias(tpr, fpr, expected):
    assert c_bias(tpr, fpr) == pytest.approx(expected, abs=1e-3)


def test_c_bias_extreme():
    assert np.isposinf(c_bias(0.0, 0.0))  # never responds, maximally conservative
    assert np.isneginf(c_bias(1.0, 1.0))  # always responds, maximally liberal


@pytest.mark.parametrize(
    "tpr,fpr,expected",
    [
        (0.5, 0.5, 1.0),  # unbiased
        (0.95, 0.20, 0.368),  # liberal
        (0.90, 0.05, 1.702),  # conservative
    ],
)
def test_beta(tpr, fpr, expected):
    assert beta(tpr, fpr) == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("tpr,fpr", [(1.0, 0.0), (0.0, 1.0)])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_beta_undefined(tpr, fpr):
    assert np.isnan(beta(tpr, fpr))


@pytest.mark.parametrize(
    "tpr,fpr,expected",
    [
        (0.5, 0.5, 0.5),  # chance
        (1.0, 0.0, 1.0),  # perfect
        (0.0, 1.0, 0.0),  # worst
        (0.25, 0.75, 0.1667),  # below chance
        (0.92, 0.44, 0.8447),  # Stanislaw & Todorov (1999)
    ],
)
def test_a_prime(tpr, fpr, expected):
    assert a_prime(tpr, fpr) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "tpr,fpr,expected",
    [
        (0.5, 0.5, 0.0),  # unbiased
        (0.4, 0.1, 0.4545),  # conservative
        (0.8, 0.4, -0.2),  # liberal
    ],
)
def test_beta_doubleprime(tpr, fpr, expected):
    assert beta_doubleprime(tpr, fpr) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("tpr,fpr", [(1.0, 0.0), (0.0, 1.0)])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_beta_doubleprime_undefined(tpr, fpr):
    assert np.isnan(beta_doubleprime(tpr, fpr))


@pytest.mark.parametrize(
    "z_intercept,z_slope,expected",
    [
        (0.0, 1.0, 0.5),  # no sensitivity
        (1.0, 1.0, 0.7602),  # equal variance (slope=1)
        (1.0, 0.5, 0.8145),  # unequal variance (slope!=1)
        (-1.0, 1.0, 0.2398),  # below chance
    ],
)
def test_a_z(z_intercept, z_slope, expected):
    assert a_z(z_intercept, z_slope) == pytest.approx(expected, abs=1e-3)


def test_scalar_inputs_return_float():
    assert isinstance(d_prime(0.75, 0.25), float)
    assert isinstance(c_bias(0.75, 0.25), float)
    assert isinstance(a_prime(0.75, 0.25), float)
    assert isinstance(beta(0.75, 0.25), float)
    assert isinstance(beta_doubleprime(0.75, 0.25), float)
    assert isinstance(a_z(1.0, 1.0), float)


def test_compute_performance_scalar():
    result = compute_performance(0.75, 0.25, z_intercept=1.0, z_slope=1.0)
    assert isinstance(result, Performance)
    assert result.d_prime == pytest.approx(1.3490, abs=1e-3)
    assert result.a_prime == pytest.approx(0.8333, abs=1e-3)
    assert result.c_bias == pytest.approx(0.0, abs=1e-6)
    assert result.beta == pytest.approx(1.0, abs=1e-6)
    assert result.a_z == pytest.approx(0.7602, abs=1e-3)


def test_compute_performance_array():
    H = np.array([0.36, 0.60, 0.76, 0.92, 0.99])
    F = np.array([0.02, 0.28, 0.40, 0.44, 0.68])
    result = compute_performance(H, F)
    # Reference values from Stanislaw & Todorov (1999)
    assert result.d_prime == pytest.approx([1.70, 0.84, 0.96, 1.56, 1.86], abs=0.01)
    assert result.c_bias == pytest.approx([1.21, 0.16, -0.23, -0.63, -1.40], abs=0.01)
    assert result.beta == pytest.approx([7.73, 1.15, 0.80, 0.38, 0.07], abs=0.01)
    # NOTE: a_prime values published in S&T appear to be incorrect for the first 3 rows.
    assert result.a_prime == pytest.approx(
        [0.8228, 0.7444, 0.7684, 0.8447, 0.8205], abs=0.001
    )
