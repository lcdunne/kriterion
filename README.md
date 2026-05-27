# Kriterion

![Kriterion logo](images/logo/library_logo_fixed.png)

**Kriterion** is a Python library for analysing data using [signal detection theory](https://en.wikipedia.org/wiki/Detection_theory).

Key features:

- Compute sensitivity and bias measures: $d'$, $c$ and more
- Fit detection models to ROC data
- Assess model fits

## Installation

It is recommend to use a virtual environment when installing python packages ([see here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)). Then:

```sh
python -m pip install kriterion
```

## Dependencies

- NumPy
- SciPy

## Usage

## Example 1: Basic signal detection theory measures

With a single true positive and false positive rate, return all common detection measures:

```python
from kriterion.measures import compute_performance


result = compute_performance(tpr=0.75, fpr=0.21)
print(result)
```

```python title="Out"
Performance(
    tpr=0.75,
    fpr=0.21,
    d_prime=1.480910997214322,
    a_prime=0.850886075949367,
    c_bias=0.06596574841107933,
    beta=1.1026202605581668,
    a_z=None
)
```

## Example 2: Receiver operating characteristic (ROC) modelling

Given a set of responses to signal and noise trials, we can use the `ROCData` class to store the raw count data, along with the cumulative ROC and z-ROC data (see the API reference for full details).

```python
from kriterion.data import ROCData


data = ROCData(
    # Strongest "signal" <---> Strongest "noise"
    # All responses to signal-present trials
    signal=[505, 248, 226, 172, 144, 93],
    # All responses to signal-absent (i.e. noise) trials
    noise=[115, 185, 304, 523, 551, 397],
)
```

To fit a model:

```python
from kriterion.fit import fit
from kriterion.models import  UnequalSignalDetection

uvsdt = UnequalSignalDetection(data)

result = fit(uvsdt)
print(uvsdt.parameters)

```

```python title="Out"
{
    'd': 1.1830254066861041,
    'signal_sd': 1.337287925732202,
    'c0': 1.0405303717702958,
    'c1': 0.46634923592441596,
    'c2': -0.06932116955166004,
    'c3': -0.6973808897916125,
    'c4': -1.4561271120010804
}
"""
```

The result of the model fitting procedure can also be shown:

```python
print(result)
```

```python title="Out"

ModelSummary(
    dof=3,
    chi2=9.183606301259807,
    chi2_p=0.02694676677704899,
    g2=9.305614752213955,
    g2_p=0.02549179488508846,
    log_likelihood=-5761.067476662813,
    aic=11536.134953325625,
    bic=11579.184187136441,
    sse=0.0004422615018773785
)
```

Finally, we can view the ROC data and the fitted model. Directly accessing the `data.*_proportions` will give the observed data, and calling the `roc()` method on the fitted model will return the curve for all criterion levels:

```python
fig, ax = plt.subplots()

ax.axis("square")
ax.plot([0, 1], [0, 1], ls="dashed", c="grey")  # Chance line

# Plots the observations
ax.scatter(
    data.noise_proportions,
    data.signal_proportions,
    c="k"
)

# Obtain the curve by calling roc()
ax.plot(*uvsdt.roc(), label="UVSDT")

ax.set(
    title='ROC with Unequal Variance Model',
    xlim=(0, 1), ylim=(0, 1),
    xlabel='FPR', ylabel='TPR',
)
ax.legend()

plt.show()

```

![roc-zroc](images/examples/uvsdt_fit.png)

## License

This project is licensed under the terms of the GPL-3.0 license.
