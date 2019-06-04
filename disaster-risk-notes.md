### Regressions using D-disaster-measure on SECID level

By itself the regression when the right hand side variable is **D** don't have a particular meaning, however, they are stil interesting. In particular, I run the following:

$$ r_{i,t+1} = \alpha+\beta D_{i,t} + \varepsilon_{i,t+1} $$

where $t$ denotes a month, $r_{i,t+1}$ is simple return during month $t+1$ of company $i$ and $D_{i,t}$ is the average 30-day D measure during month $t$. I estimated this regressions in two samples:

1. Full sample that includes all companies in the optionmetrics universe with a valid clamped V and IV
2. Subsample of firms that for which at least 80% of months in period 1996-01-01 to 2017-12-31 are sufficiently populated. A sufficiently populated month for a particular secid is defined as a month that has at least 15 nonmissing days with data.

I report the regression for both of these subsamples in a regression below where I also calculate both non-robust and robust standard errors.

```
   sample    se_type      coef        se       N
0     all  nonrobust -0.005330  0.001352  279416
1     all     robust -0.005330  0.005634  279416
2  filter  nonrobust  0.050392  0.015685   46099
3  filter     robust  0.050392  0.042208   46099
```

The overall result is quite poor. For the **all** sample the result is negative and significant only with non-robust standard errors. For smaller subsample (that I denot **filtered** subsample) the coefficient is positive. In all cases, $R^2$ is very small and less than $0.001$ which is not surprising given that it's an individual company level regressions.

### Regressions using D-disaster-measure on index level

Here I run the same type of regression but use the **aggregated** data instead of individual/secid level data. In order to do this I

1. Calculated aggregated D in two ways
   1. For each month, I calculated cross-sectional average of clamped-D $$ \overline{D} = \sum_{day \in month}\sum_{i} D_{i,day} $$
   2. I used the same tyoe of filter as I described in the previous section (at least 15 days of observations in a month and present in 80% of months). Then I extracted first principal component of  this series.
2. Estimated times series predictive regression $r_{t+1} = \alpha +\beta \overline{D}_t + u_{t+1}$ where the left hand side is either S&P 500 return from CRSP or value weighted return of all companies in CRSP.

Average **D** is sensitive to outliers (calculation of even clamped D is quite noise). Below I plot several series: (1) raw mean, (2) mean where I remove observations $<1\%$ and $>99\%$ quantiles, (3) mean where I remove observations $<0.01\%$ and $>99.99\%$ quantiles and (4) median across firms for each month.![](/Users/rsigalov/Documents/PhD/disaster-risk-revision/images/compare_truncation_of_mean_and_median.png)

The series "0.0" has several significant "drawdowns" in the later part of the sample. With truncation at 1% level it is possible to eliminate these features while preserving the dynamics of the series throughout the rest of the sample period. Hence, it seems that truncation at 1% seems to be optimal. The median series also has the same form and the correlation between 1% truncation and the median series is 0.983.