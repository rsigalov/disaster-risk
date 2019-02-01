using DataFrames
using CSV
using Dates
using NLopt
using Plots
using PyPlot
using SparseArrays
using LinearAlgebra
using Distributions
using HCubature
using ForwardDiff
include("funcs.jl")

################################################################
# Loading data
# df = CSV.read("data/opt_data.csv"; datarow = 2, delim = ",")
# df = CSV.read("data/opt_data_2.csv"; datarow = 2, delim = ",")
df = CSV.read("data/opt_data_lehman.csv"; datarow = 2, delim = ",")
df_unique = unique(df[:, [:secid,:date,:exdate]])
df_unique = sort(df_unique, [:date, :exdate])

# Loading data on interest rate to interpolate cont-compounded rate:
# zcb = CSV.read("data/zcb_cont_comp_rate.csv"; datarow = 2, delim = ",")
zcb = CSV.read("data/zcb_rates_lehman.csv"; datarow = 2, delim = ",")
zcb = sort(zcb, [:date, :days])

# spx_div_yield = CSV.read("data/spx_dividend_yield.csv"; datarow = 2, delim = ",")
spx_div_yield = CSV.read("data/div_yield_lehman.csv"; datarow = 2, delim = ",")
spx_div_yield = sort(spx_div_yield, [:secid, :date])

################################################################
# Filling an array with option data
max_options = 10
option_arr = Array{OptionData, 1}(undef, max_options)

for i_option = 1:max_options
    obs_date = df_unique[:date][i_option]
    exp_date = df_unique[:exdate][i_option]
    secid = df_unique[:secid][i_option]

    df_sub = df[(df.date .== obs_date) .& (df.exdate .== exp_date) .& (df.secid .== secid), :]
    df_sub = sort!(df_sub, :strike_price)

    df_sub = unique(df_sub)

    strikes = df_sub[:strike_price]./1000
    impl_vol = df_sub[:impl_volatility]
    spot = df_sub[:under_price][1]
    opt_days_maturity = Dates.value(exp_date - obs_date)
    T = (opt_days_maturity-1)/365 # It seems that you need to subtract 1 day
                                  # because the settlement is before the end
                                  # of the day

    ############################################################
    # Using data on ZCBs to inetrpolate risk-free rate:
    int_rate = interpolate_int_rate(obs_date, exp_date, zcb)

    ############################################################
    # In order to calculate forward price fro indices for which we
    # can assume continuously compounded dividends we need to use
    # WRDS values of dividend yield for the index.
    div_yield_cur = spx_div_yield[(spx_div_yield.date .== obs_date) .&
                                  (spx_div_yield.secid .== secid), :rate][1]/100

    forward = exp(-div_yield_cur*T)*spot/exp(-int_rate*T)

    option_arr[i_option] = OptionData(secid, obs_date, exp_date, spot, strikes,
                                      impl_vol, T, int_rate, forward)
end

################################################################
# Comparing calculated and actual call and put prices
# 1. Getting prices and strikes of specified options:
i_option = 1
for i_option = 1:10
    option = option_arr[i_option]
    obs_date = option.date
    exp_date = option.exdate
    secid = option.secid

    df_sub = df[(df.date .== obs_date) .& (df.exdate .== exp_date) .& (df.secid .== secid), :]
    df_sub = sort!(df_sub, :strike_price)
    df_sub_puts = df_sub[df_sub.cp_flag .== "P", :]
    df_sub_calls = df_sub[df_sub.cp_flag .== "C", :]

    strikes_puts = df_sub_puts[:strike_price]/1000
    prices_puts = df_sub_puts[:mid_price]
    impl_vol_puts = df_sub_puts[:impl_volatility]

    strikes_calls = df_sub_calls[:strike_price]/1000
    prices_calls = df_sub_calls[:mid_price]
    impl_vol_calls = df_sub_calls[:impl_volatility]

    r = option.int_rate
    F = option.forward
    T = option.T

    calc_prices_puts = BS_put_price.(F * exp(-r*T), 0, r,
                                     strikes_puts, impl_vol_puts, T)

    calc_prices_calls = BS_call_price.(F * exp(-r*T), 0, r,
                                       strikes_calls, impl_vol_calls, T)

    # Plotting figure with comparison
    clf()
    cla()
    fig = figure("An example", figsize=(10,8));
    ax = fig[:add_subplot](1,1,1);

    ax[:scatter]([strikes_puts; strikes_calls], [prices_puts; prices_calls],
                 alpha = 0.25, c = "b", label = "Actual Prices")
    ax[:scatter]([strikes_puts; strikes_calls], [calc_prices_puts; calc_prices_calls],
                 alpha = 0.25, c = "r", label = "Calculated BS Prices")

    ax[:set_title]("Actual vs Calculated Option Prices (puts on left, calls on right)")
    ax[:set_xlabel]("Strike")
    ax[:set_ylabel]("Price")
    legend()
    filename = string("images/compare_actual_and_calculated_prices/mid_price_BS_index_",i_option,".pdf")
    PyPlot.savefig(filename, format = "pdf",x_inches = "tight");
end

################################################################
# Testing functions for calculating Call/Put Option prices
# given a strike and interpolated volatility smile
option = option_arr[1]

svi_params_1 = fit_svi_bdbg_smile_grid(option)
svi_params_2 = fit_svi_bdbg_smile_global(option)
svi_params_3 = fit_svi_var_rho_smile_grid(option)
svi_params_4 = fit_svi_var_rho_smile_global(option)
spline_params = fitCubicSpline(option)

test_strike = 1725.0

impl_vol_1 = calc_interp_impl_vol(option, svi_params_1, test_strike)
impl_vol_2 = calc_interp_impl_vol(option, svi_params_2, test_strike)
impl_vol_3 = calc_interp_impl_vol(option, svi_params_3, test_strike)
impl_vol_4 = calc_interp_impl_vol(option, svi_params_4, test_strike)
impl_vol_5 = calc_interp_impl_vol(option, spline_params, test_strike)

call_price_1 = calc_option_value(option, svi_params_1, test_strike, "Call")
call_price_2 = calc_option_value(option, svi_params_2, test_strike, "Call")
call_price_3 = calc_option_value(option, svi_params_3, test_strike, "Call")
call_price_4 = calc_option_value(option, svi_params_4, test_strike, "Call")
call_price_5 = calc_option_value(option, spline_params, test_strike, "Call")

put_price_1 = calc_option_value(option, svi_params_1, test_strike, "Put")
put_price_2 = calc_option_value(option, svi_params_2, test_strike, "Put")
put_price_3 = calc_option_value(option, svi_params_3, test_strike, "Put")
put_price_4 = calc_option_value(option, svi_params_4, test_strike, "Put")
put_price_5 = calc_option_value(option, spline_params, test_strike, "Put")

####################################################
# Plotting different figures:
option = option_arr[1]

# Fitting volatility smiles
svi_params_1 = fit_svi_bdbg_smile_grid(option)
svi_params_2 = fit_svi_bdbg_smile_global(option)
svi_params_3 = fit_svi_var_rho_smile_grid(option)
svi_params_4 = fit_svi_var_rho_smile_global(option)
spline_params = fitCubicSpline(option)

# Setting up plot range and in-/out-of-sample plot ranges:
NTM_dist = 0.05
NTM_index = (option.strikes .<= option.spot*(1.0 + NTM_dist)) .&
            (option.strikes .>= option.spot*(1.0 - NTM_dist))
sigma_NTM = mean(option.impl_vol[NTM_index])

spot = option.spot
r = option.int_rate
T = option.T
# Plot for range [S0(1-5sigma), S0(1+5sigma)]
strike_range = LinRange(maximum([spot*(1 - 5*sigma_NTM), 100]), spot*(1 + 2*sigma_NTM), 500)
in_sample_index = (strike_range .>= minimum(option.strikes)) .&
                  (strike_range .<= maximum(option.strikes))
below_sample_index = strike_range .< minimum(option.strikes)
above_sample_index = strike_range .> maximum(option.strikes)
in_sample_below_spot_index = (strike_range .<= option.spot) .&
                       (strike_range .>= minimum(option.strikes))
in_sample_above_spot_index = (strike_range .> option.spot) .&
                             (strike_range .<= maximum(option.strikes))

strike_range_ins = strike_range[in_sample_index] # Strikes "in sample"
strike_range_below = strike_range[below_sample_index] # Strikes "below" the sample
strike_range_above = strike_range[above_sample_index] # Strikes "below" the sample
strike_range_ins_below_spot = strike_range[in_sample_below_spot_index]
strike_range_ins_above_spot = strike_range[in_sample_above_spot_index]

cla()
clf()
fig = figure("An example", figsize=(16,12));
PyPlot.axis("off")

ax1 = fig[:add_subplot](2,2,1);
ax2 = fig[:add_subplot](2,2,2);
ax3 = fig[:add_subplot](2,2,3);
ax4 = fig[:add_subplot](2,2,4);

param_list = [svi_params_1, svi_params_2, svi_params_3, svi_params_4] #, spline_params]
param_name_list = ["Zero Rho, Grid", "Zero Rho, Global", "Variable Rho, Grid", "Variable Rho, Global", "Cubic Spline"]
color_list = ["k", "r", "b", "g", "c"]

# Adding implied volatility from OptionMetrics scatter plot:
ax1[:scatter](option.strikes, option.impl_vol, alpha = 0.1, c = "b")

for i = 1:length(param_list)
    # 1. Implied volatilities
    print(i)
    interp_vol_lambda = K -> calc_interp_impl_vol(option, param_list[i], K)
    ax1[:plot](strike_range_ins, map(interp_vol_lambda, strike_range_ins), alpha = 0.75, linestyle = "-", c = color_list[i], label = param_name_list[i])
    ax1[:plot](strike_range_below, map(interp_vol_lambda, strike_range_below), alpha = 0.75, linestyle = "--", c = color_list[i])
    ax1[:plot](strike_range_above, map(interp_vol_lambda, strike_range_above), alpha = 0.75, linestyle = "--", c = color_list[i])
    ax1[:axvline](x = option.spot)

    ax1[:set_title]("Interpolated Volatility (sigma)")
    ax1[:set_xlabel]("Strike")
    ax1[:set_ylabel]("")

    # 2. OTM option prices
    calc_option_value_put = K -> calc_option_value(option, param_list[i], K, "Put")
    calc_option_value_call = K -> calc_option_value(option, param_list[i], K, "Call")
    ax2[:plot](strike_range_ins_below_spot, map(calc_option_value_put, strike_range_ins_below_spot), alpha = 0.75, linestyle = "-", c = color_list[i], label = param_name_list[i])
    ax2[:plot](strike_range_ins_above_spot, map(calc_option_value_call, strike_range_ins_above_spot), alpha = 0.75, linestyle = "-", c = color_list[i])
    ax2[:plot](strike_range_below, map(calc_option_value_put, strike_range_below), alpha = 0.75, linestyle = "--", c = color_list[i], label = param_name_list[i])
    ax2[:plot](strike_range_above, map(calc_option_value_call, strike_range_above), alpha = 0.75, linestyle = "--", c = color_list[i], label = param_name_list[i])
    ax2[:axvline](x = option.spot)

    ax2[:set_title]("OTM Option Prices")
    ax2[:set_xlabel]("Strike")
    ax2[:set_ylabel]("")

    # 3. Risk-Neutral CDF
    calc_RN_CDF_lambda = K -> calc_RN_CDF_PDF(option, param_list[i], K)[1]
    ax3[:plot](strike_range_ins, map(calc_RN_CDF_lambda, strike_range_ins), alpha = 0.75, linestyle = "-", c = color_list[i], label = param_name_list[i])
    ax3[:plot](strike_range_below, map(calc_RN_CDF_lambda, strike_range_below), alpha = 0.75, linestyle = "--", c = color_list[i])
    ax3[:plot](strike_range_above, map(calc_RN_CDF_lambda, strike_range_above), alpha = 0.75, linestyle = "--", c = color_list[i])
    ax3[:axvline](x = option.spot)

    ax3[:set_title]("Risk-Neutral CDF")
    ax3[:set_xlabel]("Strike")
    ax3[:set_ylabel]("")

    # 4. Risk-Neutral PDF
    calc_RN_PDF_lambda = K -> calc_RN_CDF_PDF(option, param_list[i], K)[2]
    ax4[:plot](strike_range_ins, map(calc_RN_PDF_lambda, strike_range_ins), alpha = 0.75, linestyle = "-", c = color_list[i], label = param_name_list[i])
    ax4[:plot](strike_range_below, map(calc_RN_PDF_lambda, strike_range_below), alpha = 0.75, linestyle = "--", c = color_list[i])
    ax4[:plot](strike_range_above, map(calc_RN_PDF_lambda, strike_range_above), alpha = 0.75, linestyle = "--", c = color_list[i])
    ax4[:axvline](x = option.spot)

    ax4[:set_title]("Risk-Neutral PDF")
    ax4[:set_xlabel]("Strike")
    ax4[:set_ylabel]("")
end

title_text = string("S&P 500 option: from ", option.date, " to ", option.exdate);
suptitle(title_text);
legend()
# filename = string("images/compare_actual_and_calculated_prices/mid_price_BS_index_",i_option,".pdf")
PyPlot.savefig("example_lehman_no_spline.pdf", format = "pdf",x_inches = "tight");


################################################################
# Checking fit of SVI volatility smile model in the data

# 1. Fitting different SVI specifications:
svi_bdbg_grid_params_arr = map(fit_svi_bdbg_smile_grid, option_arr)
svi_bdbg_global_params_arr = map(fit_svi_bdbg_smile_global, option_arr)
svi_var_rho_grid_params_arr = map(fit_svi_var_rho_smile_grid, option_arr)
svi_var_rho_global_params_arr = map(fit_svi_var_rho_smile_global, option_arr)

# Plotting SVI fit with actual
for i_option = 1:10
    print(i_option)
    print("\n")

    clf()
    fig = figure("An example", figsize=(10,8));
    PyPlot.axis("off")

    ax1 = fig[:add_subplot](2,2,1);
    ax2 = fig[:add_subplot](2,2,2);
    ax3 = fig[:add_subplot](2,2,3);
    ax4 = fig[:add_subplot](2,2,4);

    plot_vol_smile(option_arr[i_option], svi_bdbg_grid_params_arr[i_option], "Rho = 0, Grid", ax1);
    plot_vol_smile(option_arr[i_option], svi_bdbg_global_params_arr[i_option], "Rho = 0, Global", ax2);
    plot_vol_smile(option_arr[i_option], svi_var_rho_grid_params_arr[i_option], "Variable Rho, Grid" , ax3);
    plot_vol_smile(option_arr[i_option], svi_var_rho_global_params_arr[i_option], "Variable Rho, Global", ax4);

    subplots_adjust(wspace = 0.35, hspace = 0.35);

    title_text = string("SP500, from ", option_arr[i_option].date, " to ", option_arr[i_option].exdate);
    suptitle(title_text);
    filepath_to_save = string("images/julia_comparison_lehman/option_",i_option ,".pdf")
    PyPlot.savefig(filepath_to_save, format="pdf", bbox_inches= "tight");
end



# ################################################################
# # Functions to calculate integrands required for calculating
# # integrals for V and IV
# ################################################################
#
# spot = option.spot
#
# # function to calculate call option price for a specific
# # option and interpolation parameters:
# function calc_specific_option_call_value(K)
#     return calc_option_value(option, svi_params_1, K, "Call")
# end
#
# function calc_specific_option_put_value(K)
#     return calc_option_value(option, svi_params_1, K, "Put")
# end
#
# # First integrand of V(0,T):
# function V1(K)
#     return 2 * (1 - log(K/spot)) * calc_specific_option_call_value(K)/K^2
# end
#
# # First integrand of V(0,T):
# function V2(K)
#     return 2 * (1 + log(spot/K)) * calc_specific_option_call_value(K)/K^2
# end
#
# ################################################################
#
# function f(x)
#     return cos(x)
# end
#
# hquadrature(f, -1, 100)
