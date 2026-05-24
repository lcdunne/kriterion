# Objective Functions

The objective functions are used to find the best fit for the chosen model. Several functions are supported, including the $\chi^2$, $G^2$, Log-likelihood, and $\text{SSE}$.

There are tradeoffs with each approach. Those with the `_binom` suffix use the cumulative data, and treat the data at each criterion as a binomial with the counts above vs below the threshold, and then summing across all criteria. These variants converge reliably and are not impacted by sparse cell counts. However, these violate independence; because it works on cumulative data, successive threshold levels contain data from preceding levels. It treats overlapping data as if they were separate observations.

Those without the `_binom` suffix compute the test statistic on the non-cumulative counts. This is more statistically consistent, but is sensitive to sparse cell counts (where an observer makes few or no responses at a particular criterion) which can make the $\chi^2$, $G^2$, and likelihood ratio approximations unreliable. Pooling responses across observers mitigates this, though it precludes single-subject analyses.

A middle-ground for these tradeoffs might be to use the binomial approach to simply find the best estimates for the parameters, but to perform model comparisons and significance testing using the non-binomial variants.

See [^cressie_read_1984] for reference.

[^cressie_read_1984]: [Cressie, N., & Timothy R. C. Read. (1984). Multinomial Goodness-of-Fit Tests. Journal of the Royal Statistical Society. Series B (Methodological), 46(3), 440–464](http://www.jstor.org/stable/2345686).

::: kriterion.objectives
