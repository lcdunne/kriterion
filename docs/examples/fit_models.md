# Fitting Theoretical Models

## 1. Setup

Given a set of responses to signal and noise trials, we can use the `ROCData` class to store the raw count data. This will then generate the cumulative ROC and z-ROC data (see the [API reference](../api/data_helpers.md#kriterion.data.ROCData) for full details). All models require an instance of `ROCData`:

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

## 2. Fit

To fit a model, call the `fit` function and pass in the model. By default, `fit` will seek to minimise the log-likelihood, but this can be changed to use any other [objective function](../api/objective_functions.md). This will modify the model's parameters and return a fit result containing goodness of fit statistics:

```python
from kriterion.fit import fit
from kriterion.models import  UnequalSignalDetection


uvsdt = UnequalSignalDetection(data)

result = fit(uvsdt)

print(result)
print(uvsdt.parameters)

```

```python title="Out - Goodness of Fit statistics"

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

```python title="Out - Fitted model parameters"
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

The fitted model parameters now represent the maximum likelihood estimates.

## 3. Visualise

We can then view the ROC data and the fitted model. Directly accessing the `data.*_proportions` will give the observed data, and calling the `roc()` method on the fitted model will return the curve for all criterion levels:

```python
import matplotlib.pyplot as plt


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

![roc-zroc](./images/examples/uvsd_roc_light.png#only-light)
![roc-zroc](./images/examples/uvsd_roc_dark.png#only-dark)
