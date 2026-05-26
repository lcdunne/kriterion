# Objective Functions

Several objective functions are supported for fitting models: $\chi^2$, $G^2$, log-likelihood, and $\text{SSE}$.

The $\chi^2$ and $G^2$ statistics are also available in `_cumulative` variants. The standard variants compute the statistic from cell counts directly. This is conventional but it makes the procedure sensitive to sparse cells, e.g. when an observer makes few or no responses at a given criterion.

The `_cumulative` variants instead evaluate the fit via a binary split at each criterion threshold, with responses at or above the threshold versus those below, and then summing across all thresholds. This is robust to sparse cells, and is useful for getting a reliable fit during optimisation. However, successive thresholds share responses with all preceding ones, violating independence, so this is not recommended for model comparison.

See [^cressie_read_1984] for reference.

[^cressie_read_1984]: [Cressie, N., & Timothy R. C. Read. (1984). Multinomial Goodness-of-Fit Tests. Journal of the Royal Statistical Society. Series B (Methodological), 46(3), 440–464](http://www.jstor.org/stable/2345686).

::: kriterion.objectives
