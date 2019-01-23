* **julia_comparison**. Compares fit of SVI curve among several specifications and
methods of optimization: (1) Fixed rho = 0 with a grid search to find a starting
value for local optimization, (2) Fixed rho = 0 with a global optimizer to find
a starting value for local optimization, (3) and (4) the same as before but with
variable rho

* **julia_comparison_SVX_only**. Same as **julia_comparison** but excludes SVXW
(weekly SVX options). The reason is that there can be options with the same strike
and maturity but different prices and hence implied volatilities. The only difference
is that one is SVX and another is SVXW.

* **julia_spline_fit**. Plots fit of Cubic Spline Interpolation with volatility
clamped at lowest and highest strike: derivative at boundary nodes is set to zero

* **compare_actual_and_calculated_prices**. Compares mid-prices from the WRDS
with mid-prices calculated using implied volatilities from WRDS and data on
dividend yield (in case of indices) and dividend distributions (in case of
individual equity)
