

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

#### Side note on construction and comparison of aggregate D measures

Here I run the same type of regression but use the **aggregated** data instead of individual/secid level data. In order to do this I

1. Calculated aggregated D in two ways
   1. For each month, I calculated cross-sectional average of clamped-D $$ \overline{D} = \sum_{day \in month}\sum_{i} D_{i,day} $$
   2. I used the same tyoe of filter as I described in the previous section (at least 15 days of observations in a month and present in 80% of months). Then I extracted first principal component of  this series.
2. Estimated times series predictive regression $r_{t+1} = \alpha +\beta \overline{D}_t + u_{t+1}$ where the left hand side is either S&P 500 return from CRSP or value weighted return of all companies in CRSP.

Average **D** is sensitive to outliers (calculation of even clamped D is quite noise). Below I plot several series: (1) raw mean, (2) mean where I remove observations $<1\%$ and $>99\%$ quantiles, (3) mean where I remove observations $<0.01\%$ and $>99.99\%$ quantiles and (4) median across firms for each month.![](/Users/rsigalov/Documents/PhD/disaster-risk-revision/images/compare_truncation_of_mean_and_median.png)

The series "0.0" has several significant "drawdowns" in the later part of the sample. With truncation at 1% level it is possible to eliminate these features while preserving the dynamics of the series throughout the rest of the sample period. Hence, it seems that truncation at 1% seems to be optimal. The median series also has the same form and the correlation between 1% truncation and the median series is 0.983.

Next, I do the filtering described above, first, to calculate the mean on filtered series and then to extract the first principal component. On the Figure below I compare these two with a series from above where I use all companies and trim at 1% level in each month.

![](/Users/rsigalov/Documents/PhD/disaster-risk-revision/images/compare_mean_filtered_and_pc1.png)

The correlation matrix between the measures is

```
 PC1         1.0       0.99484   0.956116
 Mean Filter 0.99484   1.0       0.964916
 Mean All    0.956116  0.964916  1.0
```

The first principal component and the mean on filtered series are almost the same. And they are also very close to the mean on the full series with 1% truncation. 

#### Estimating aggregate predictive regressions

Now we can move to estimating regressions with the three series described in the previous section. Below are the results of these regressions:

```
        Y              X se_type      coef        se        R2
0  vw_ret     D_mean_all    homo -0.448365  0.217988  0.015891
0  vw_ret     D_mean_all  hetero -0.448365  0.326521  0.015891
0  vw_ret          D_pc1    homo -0.312464  0.196686  0.009541
0  vw_ret          D_pc1  hetero -0.312464  0.316954  0.009541
0  vw_ret  D_mean_filter    homo -0.342058  0.201877  0.010839
0  vw_ret  D_mean_filter  hetero -0.342058  0.323205  0.010839
0  sp_ret     D_mean_all    homo -0.461315  0.218361  0.016750
0  sp_ret     D_mean_all  hetero -0.461315  0.328866  0.016750
0  sp_ret          D_pc1    homo -0.322898  0.197049  0.010145
0  sp_ret          D_pc1  hetero -0.322898  0.319054  0.010145
0  sp_ret  D_mean_filter    homo -0.352182  0.202250  0.011441
0  sp_ret  D_mean_filter  hetero -0.352182  0.325213  0.011441
```

The coefficients are now much stronger in magnitude, nevertheless, they are insignicant with robust standard errors.

### Probability Model Using Risk-Neutral Probability of Disaster

#### Individual secid regressions

I this part we use the interpolated series of a certain percent monthly decline to forecast declines on both individual and aggregate levels. In particular, we estimate the following model

$$ \mathbb{I}\{ret_{i,t+1} \le -20\%\} = \alpha + \beta \cdot \mathbb{P}^{\mathbb{Q}}_t(ret_{i,t+1} \le -20\%) + \varepsilon_{i,t+1} $$

Since the estimated probability is risk neutral, we don't expect $\beta = 1$, however, it should be fairly close and $\beta < 1$ meaning that risk neutral probability of a decline is **overpredicting** the actual probability since it upweights bad realizations of return (*It should be priced for it to be the case, so need to think more carefully about this point*). We present the results for the whole sample below:

```
-----------------------------------------------------------
                              (1)          (2)          (3)
                       Base (All) Cluster (A~) No Crisis ~)
                             b/se         b/se         b/se
-----------------------------------------------------------
20% rn prob                 1.017        1.017        1.055
                         (0.0114)     (0.0937)      (0.108)
Constant                   0.0121       0.0121      0.00799
                       (0.000482)    (0.00370)    (0.00347)
-----------------------------------------------------------
R-squared                  0.0745       0.0745       0.0825
N. of obs                  279416       279416       231583
-----------------------------------------------------------
```

Column (1) features HC standard errors, column (2) double clusters by firm and month and column (3) excludes 2007, 2008 and 2009. I repeat the same exercise for the filtered secede-months and present the results below

```
-----------------------------------------------------------
                              (1)          (2)          (3)
                     Base (Filt~) Cluster (F~) No Crisis ~)
                             b/se         b/se         b/se
-----------------------------------------------------------
20% rn prob                 0.755        0.755        0.781
                         (0.0295)      (0.101)      (0.112)
Constant                -0.000593    -0.000593     -0.00238
                       (0.000868)    (0.00276)    (0.00301)
-----------------------------------------------------------
R-squared                  0.0536       0.0536       0.0557
N. of obs                   46099        46099        39236
-----------------------------------------------------------
```

I then do the same two exercises for 40% declines instead of 20% decline and get the following results. First for the whole sample

```
-----------------------------------------------------------
                              (1)          (2)          (3)
                       Base (All) Cluster (A~) No Crisis ~)
                             b/se         b/se         b/se
-----------------------------------------------------------
40% rn prob                 0.610        0.610        0.656
                         (0.0201)     (0.0917)      (0.118)
Constant                  0.00258      0.00258      0.00177
                       (0.000203)   (0.000806)   (0.000677)
-----------------------------------------------------------
R-squared                  0.0316       0.0316       0.0346
N. of obs                  279416       279416       231583
-----------------------------------------------------------
```

then for the filtered secid-months

```
-----------------------------------------------------------
                              (1)          (2)          (3)
                     Base (Filt~) Cluster (F~) No Crisis ~)
                             b/se         b/se         b/se
-----------------------------------------------------------
40% rn prob                 0.421        0.421        0.439
                         (0.0573)     (0.0997)      (0.125)
Constant               -0.0000866   -0.0000866    -0.000392
                       (0.000295)   (0.000444)   (0.000482)
-----------------------------------------------------------
R-squared                  0.0168       0.0168       0.0157
N. of obs                   46099        46099        39236
-----------------------------------------------------------
```

All these results suggest that risk-neutral probability of a certain percent decline has a predictive power at the individual secid level. 

#### Aggregate regressions

For this part I used the same methodology as for **D**. First, I calculated the cross-sectional mean of interpolated measure $\mathbb{P}_t^{Q}(r_{i,t+1} < -0.2)$ for each month to construct an aggregate series. Unlike the D series this one is not sensitive to outliers (as there are few) and, therefore, I didn't trim it. Next I filtered the data in the same way and also estimated the first principal component. First, I compare aggregate D and probability series in the Figure below

![](/Users/rsigalov/Documents/PhD/disaster-risk-revision/images/compare_agg_D_and_prob.png)

Correlation between all aggregate probability measures is very high and the correlation between the aggregate D and probability measures is 0.82 on average. The most noticeable difference between the two measures is dot-com bubble. Whereas D stayed elevated, probability measure exploded to near Great Recession levels. Also D measure stayed very close to zero during the expansion of 2000s.

Since the aggregate stock market doesn't fall significantly that often: in fact in this sample there is only one month — October 2008 — when the return on the S&P 500 was less than -15% and only four months when the return was less than -10%. Hence, there is not enough variation to run the regression

$$ \mathbb{I}\{ret_{m,t+1} \le -20\%\} = \alpha + \beta \cdot \overline{\mathbb{P}^{\mathbb{Q}}_t(ret_{i,t+1} \le -20\%)} + \varepsilon_{t+1} $$

For that reason I estimated a series of the following regressions

$$ \mathbb{I}\{ret_{m,t+1} \le x\} = \alpha + \beta^{(x)} \cdot \overline{\mathbb{P}^{\mathbb{Q}}_t(ret_{i,t+1} \le -20\%)} + \varepsilon_{t+1} $$

where $r_{m,t+1}$ is the return on S&P 500 and $ x \in \{-10\%,-9\%,\dots,-1\%, 0\%\}$. I then plot coefficient $\beta^{(x)}$ along with 95% confidence bands as a function of $x$ in the Figure below

![](/Users/rsigalov/Documents/PhD/disaster-risk-revision/images/agg_prob_forecast.png)

#### Forming trading strategy on the basis of disaster measure

Here we formed portfolios on the basis of calculated disaster measure: clamped D or the risk neutral  probability of a certain % decline in the stock price. In particular, 

1. Each month we we calculated cross sectional 30% and 70% quantile of a disaster measure.
2. We formed equal weighted portfolios of secids with disaster measure lower than 30% quantile and secids with disaster measure greater than 70% quantile.
3. We long the portfolio with low disaster measure and short the portfolio with high disaster measure.











































