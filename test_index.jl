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
df = CSV.read("data/opt_data_3.csv"; datarow = 2, delim = ",")
# df = CSV.read("data/opt_data_2008.csv"; datarow = 2, delim = ",")
# df = CSV.read("data/opt_data_lehman.csv"; datarow = 2, delim = ",")
df_unique = unique(df[:, [:secid,:date,:exdate]])
df_unique = sort(df_unique, [:date, :exdate])

# Loading data on interest rate to interpolate cont-compounded rate:
zcb = CSV.read("data/zcb_cont_comp_rate.csv"; datarow = 2, delim = ",")
# zcb = CSV.read("data/zcb_cont_comp_rate_2008.csv"; datarow = 2, delim = ",")
# zcb = CSV.read("data/zcb_rates_lehman.csv"; datarow = 2, delim = ",")
zcb = sort(zcb, [:date, :days])

spx_div_yield = CSV.read("data/spx_dividend_yield.csv"; datarow = 2, delim = ",")
# spx_div_yield = CSV.read("data/spx_dividend_yield_2008.csv"; datarow = 2, delim = ",")
# spx_div_yield = CSV.read("data/div_yield_lehman.csv"; datarow = 2, delim = ",")
spx_div_yield = sort(spx_div_yield, [:secid, :date])

################################################################
# Filling an array with option data
max_options = size(df_unique)[1]
option_arr = Array{OptionData, 1}(undef, max_options)

for i_option = 1:max_options
    print(i_option)
    print("\n")
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
    T = (opt_days_maturity - 1)/365 # It seems that you need to subtract 1 day
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
option = option_arr[31]

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
    legend()

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
    legend()

    # 3. Risk-Neutral CDF
    calc_RN_CDF_lambda = K -> calc_RN_CDF_PDF(option, param_list[i], K)[1]
    ax3[:plot](strike_range_ins, map(calc_RN_CDF_lambda, strike_range_ins), alpha = 0.75, linestyle = "-", c = color_list[i], label = param_name_list[i])
    ax3[:plot](strike_range_below, map(calc_RN_CDF_lambda, strike_range_below), alpha = 0.75, linestyle = "--", c = color_list[i])
    ax3[:plot](strike_range_above, map(calc_RN_CDF_lambda, strike_range_above), alpha = 0.75, linestyle = "--", c = color_list[i])
    ax3[:axvline](x = option.spot)

    ax3[:set_title]("Risk-Neutral CDF")
    ax3[:set_xlabel]("Strike")
    ax3[:set_ylabel]("")
    legend()

    # 4. Risk-Neutral PDF
    calc_RN_PDF_lambda = K -> calc_RN_CDF_PDF(option, param_list[i], K)[2]
    ax4[:plot](strike_range_ins, map(calc_RN_PDF_lambda, strike_range_ins), alpha = 0.75, linestyle = "-", c = color_list[i], label = param_name_list[i])
    ax4[:plot](strike_range_below, map(calc_RN_PDF_lambda, strike_range_below), alpha = 0.75, linestyle = "--", c = color_list[i])
    ax4[:plot](strike_range_above, map(calc_RN_PDF_lambda, strike_range_above), alpha = 0.75, linestyle = "--", c = color_list[i])
    ax4[:axvline](x = option.spot)

    ax4[:set_title]("Risk-Neutral PDF")
    ax4[:set_xlabel]("Strike")
    ax4[:set_ylabel]("")
    legend()
end

title_text = string("S&P 500 option: from ", option.date, " to ", option.exdate);
suptitle(title_text);
# legend()
# filename = string("images/compare_actual_and_calculated_prices/mid_price_BS_index_",i_option,".pdf")
PyPlot.savefig("example_2017.pdf", format = "pdf", x_inches = "tight");

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

################################################################
# Functions to calculate integrands required for calculating
# integrals for V and IV
################################################################

function calc_variation_and_jump_risk(option::OptionData, interp_params)
    spot = option.spot
    r = option.int_rate
    T = option.T

    # function to calculate call option price for a specific
    # option and interpolation parameters:
    calc_option_value_put = K -> calc_option_value(option, interp_params, K, "Put")
    calc_option_value_call = K -> calc_option_value(option, interp_params, K, "Call")

    # I will make a change of variables to integrate all variables from 0 to 1.
    # This is necessary for inifinite limit integrals but not necessary for
    # integrals with finite limit. Nevertheless, to make everything similar
    # I will do a change of variables for all integrals.

    # First integrxnd of V(0, T) -- value of a quadratic claim:
    V1_raw = K -> 2 * (1 - log(K/spot)) * calc_option_value_call(K)/K^2
    V1 = t -> V1_raw(spot + t/(1-t))/(1-t)^2

    # Second integrand of V(0, T) -- value of a quadratic claim:
    V2_raw = K -> 2 * (1 + log(spot/K)) * calc_option_value_put(K)/K^2
    V2 = t -> V2_raw(spot * t) * spot

    # First Integrand of W(0, T) -- value of a cubic claim
    W1_raw = K -> (6 * log(K/spot) - 3 * (log(K/spot))^2) * calc_option_value_call(K)/K^2
    W1 = t -> W1_raw(spot + t/(1-t))/(1-t)^2

    # Second Integrand of W(0, T) -- value of a cubic claim
    W2_raw = K -> (6 * log(spot/K) + 3 * log(spot/K)^2) * calc_option_value_put(K)/K^2
    W2 = t -> W2_raw(spot * t) * spot

    # First Integrand of X(0, T) -- value of a quatric claim
    X1_raw = K -> (12 * log(K/spot)^2 - 4 * log(K/spot)^3) * calc_option_value_call(K)/K^2
    X1 = t -> X1_raw(spot + t/(1-t))/(1-t)^2

    # Second Integrand of X(0, T) -- value of a quatric claim
    X2_raw = K -> (12 * log(spot/K)^2 + 4 * log(spot/K)^3) * calc_option_value_put(K)/K^2
    X2 = t -> X2_raw(spot * t) * spot

    # First Integrand of e^{-rT}IV
    IV1_raw = K -> calc_option_value_call(K)/K^2
    IV1 = t -> IV1_raw(spot + t/(1-t))/(1-t)^2

    # Second Integrand of e^{-rT}IV
    IV2_raw = K -> calc_option_value_put(K)/K^2
    IV2 = t -> IV2_raw(spot * t) * spot

    V = hquadrature(V1, 0, 1)[1] + hquadrature(V2, 0, 1)[1]
    W = hquadrature(W1, 0, 1)[1] + hquadrature(W2, 0, 1)[1]
    X = hquadrature(X1, 0, 1)[1] + hquadrature(X2, 0, 1)[1]

    mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

    variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)

    integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1)[1] + hquadrature(IV2, 0, 1)[1] - exp(-r*T)*(exp(r*T)-1-r*T))

    D = variation - integrated_variation

    return variation, integrated_variation, D
end

option = option_arr[3]

svi_params_1 = fit_svi_bdbg_smile_grid(option)
svi_params_2 = fit_svi_bdbg_smile_global(option)
svi_params_3 = fit_svi_var_rho_smile_grid(option)
svi_params_4 = fit_svi_var_rho_smile_global(option)
spline_params = fitCubicSpline(option)

param_list = [svi_params_1, svi_params_2, svi_params_3, svi_params_4, spline_params]

calc_variation_and_jump_risk(option, param_list[1])
calc_variation_and_jump_risk(option, param_list[2])
calc_variation_and_jump_risk(option, param_list[3])
calc_variation_and_jump_risk(option, param_list[4])
calc_variation_and_jump_risk(option, param_list[5])

############################################################
# Generating a bunch of characteristics for many options
var_chars = DataFrame(date = [], exdate = [], fit_type = [], sigmaNTM = [], prob_40_perc = [],
                       prob_2sigma = [], prob_5sigma = [], put_2sigma = [],
                       put_5sigma = [], V = [], IV = [], D = [], high_CDF = [])
fit_type_list = ["rho=0, grid", "rho=0, global", "VarRho, grid", "VarRho, global", "Spline"]

for i = 30:35
    print("Evaluating option ")
    print(i)
    print("\n")
    option = option_arr[i]
    spot = option.spot

    # Fitting different volatility smile functions
    svi_params_1 = fit_svi_bdbg_smile_grid(option)
    svi_params_2 = fit_svi_bdbg_smile_global(option)
    svi_params_3 = fit_svi_var_rho_smile_grid(option)
    svi_params_4 = fit_svi_var_rho_smile_global(option)
    spline_params = fitCubicSpline(option)

    param_list = [svi_params_1, svi_params_2, svi_params_3, svi_params_4, spline_params]

    NTM_dist = 0.05
    NTM_index = (option.strikes .<= option.spot*(1.0 + NTM_dist)) .&
                (option.strikes .>= option.spot*(1.0 - NTM_dist))
    sigma_NTM = mean(option.impl_vol[NTM_index])

    for j = 1:length(fit_type_list)
        calc_RN_CDF_lambda = K -> calc_RN_CDF_PDF(option, param_list[j], K)[1]
        prob_40_perc = calc_RN_CDF_lambda(spot * (1 - 0.4))
        prob_2sigma = spot*(1 - 2*sigma_NTM) > 0 ? calc_RN_CDF_lambda(spot * (1 - 2 * sigma_NTM)) : missing
        prob_5sigma = spot*(1 - 5*sigma_NTM) > 0 ? calc_RN_CDF_lambda(spot * (1 - 5 * sigma_NTM)) : missing

        calc_option_value_put = K -> calc_option_value(option, param_list[j], K, "Put")
        put_2sigma = spot*(1 - 2*sigma_NTM) > 0 ? calc_option_value_put(spot * (1 - 2*sigma_NTM)) : missing
        put_5sigma = spot*(1 - 5*sigma_NTM) > 0 ? calc_option_value_put(spot * (1 - 5*sigma_NTM)) : missing

        V, IV, D = calc_variation_and_jump_risk(option, param_list[j])

        # Filling the table:
        to_append = DataFrame(date = [option.date], exdate = [option.exdate],
            fit_type = [fit_type_list[j]], sigmaNTM = [sigma_NTM],
            prob_40_perc = [prob_40_perc],
            prob_2sigma = [prob_2sigma], prob_5sigma = [prob_5sigma],
            put_2sigma = [put_2sigma], put_5sigma = [put_5sigma],
            V = [V], IV = [IV], D = [D], high_CDF = [calc_RN_CDF_lambda(3000)])

        append!(var_chars, to_append)
    end
end

CSV.write("output/chars_2.csv",  var_chars)


################################################################
# Calculating VIX using the methodology from their whitepaper
# https://www.cboe.com/micro/vix/vixwhite.pdf
################################################################
# 1. Comparing actual mid-prices and BS calculated prices using
# reported implied volatility
# option = option_arr[13]
# r = option.int_rate
# F = option.forward
# T = option.T
#
# svi_params_1 = fit_svi_bdbg_smile_grid(option)
# svi_params_2 = fit_svi_bdbg_smile_global(option)
# svi_params_3 = fit_svi_var_rho_smile_grid(option)
# svi_params_4 = fit_svi_var_rho_smile_global(option)
# interp_param_list = [svi_params_1, svi_params_2, svi_params_3, svi_params_4]
#
# calc_option_value_put = K -> calc_option_value(option, interp_params, K, "Put")
# calc_option_value_call = K -> calc_option_value(option, interp_params, K, "Call")
#
# strikes_puts = option.strikes[option.strikes .<= option.spot]
# strikes_calls = option.strikes[option.strikes .> option.spot]
#
# impl_vol_puts = option.impl_vol[option.strikes .<= option.spot]
# impl_vol_calls = option.impl_vol[option.strikes .> option.spot]
#
# # 1. calculating prices for each strikes and implied volatility:
# calc_prices_puts = BS_put_price.(F * exp(-r*T), 0, r, strikes_puts, impl_vol_puts, T)
# calc_prices_calls = BS_call_price.(F * exp(-r*T), 0, r, strikes_calls, impl_vol_calls, T)
#
# df_sub = df[(df.date .== option.date) .&
#                  (df.exdate .== option.exdate) .&
#                  (df.secid .== option.secid), :]
# df_sub_puts = sort(df_sub[df_sub.cp_flag .== "P", :], :strike_price)
# df_sub_calls = sort(df_sub[df_sub.cp_flag .== "C", :], :strike_price)
#
# actual_prices_puts = df_sub_puts[:mid_price]
# actual_prices_calls = df_sub_calls[:mid_price]

# 3. Calculating integral only "in-sample", i.e. the limits of integration are
# from the first to the last strike
# calc_option_value_put = K -> calc_option_value(option, interp_param_list[1], K, "Put")
# calc_option_value_call = K -> calc_option_value(option, interp_param_list[1], K, "Call")
#
# IV1_raw = K -> calc_option_value_call(K)/K^2
# IV2_raw = K -> calc_option_value_put(K)/K^2
#
# spot = option.spot
# min_K = minimum(strikes)
# max_K = maximum(strikes)
# IV_limit = (2*exp(r*T)/T) * (hquadrature(IV1_raw, spot, max_K)[1] + hquadrature(IV2_raw, min_K, spot)[1]) - (2/T)*(exp(r*T)-1-r*T)
# VIX_like_IV_limit = sqrt(IV_limit) * 100
#
# IV1 = t -> IV1_raw(spot + t/(1-t))/(1-t)^2
# IV2 = t -> IV2_raw(spot * t) * spot
# IV = (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1)[1] + hquadrature(IV2, 0, 1)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
# VIX_like_IV = sqrt(IV) * 100
#
# V, IV, D = calc_variation_and_jump_risk(option, interp_param_list[1])

# 4. Getting options with maturity close to 30 days
unique_dates = unique(df_unique[:date])
option_dates_arr = map(x -> x.date, option_arr)

# # list of the options that are closest to 30 days maturity for a given
# # oservation date:
option_arr_close_30 = Array{OptionData, 1}(undef, length(unique_dates))
for i = 1:length(unique_dates)
    current_date = unique_dates[i]
    current_option_arr = option_arr[map(x -> x.date, option_arr) .== current_date]

    dist_to_30 = abs.(map(x -> x.T*365, current_option_arr) .- 30)
    index_close_30 = findmin(dist_to_30)[2] # index of the option closest to 30 days in maturity
    option_arr_close_30[i] = current_option_arr[index_close_30]
end

option_arr_exactly_30 = option_arr[map(x -> x.T*365, option_arr) .== 30]

# For each of these options need to calculate VIX, limited IV (that should be very
# close to VIX), IV, V, D
variation_comp = DataFrame(date = [], exdate = [], fit_type = [],
                      VIX = [], IV_limit = [], IV = [], V = [], D = [])
fit_type_list = ["rho=0, grid", "rho=0, global", "VarRho, grid", "VarRho, global"]

for i = 1:length(option_arr_close_30)
    print("Option # ")
    print(i)
    print("/")
    print(length(option_arr_close_30))
    print("\n")

    option = option_arr_close_30[i]

    VIX_i = calc_VIX(option) # interpolation free measure
    spot = option.spot
    r = option.int_rate
    T = option.T
    strikes = option.strikes

    # Fitting different SVIs
    svi_params_1 = fit_svi_bdbg_smile_grid(option)
    svi_params_2 = fit_svi_bdbg_smile_global(option)
    svi_params_3 = fit_svi_var_rho_smile_grid(option)
    svi_params_4 = fit_svi_var_rho_smile_global(option)
    param_list = [svi_params_1, svi_params_2, svi_params_3, svi_params_4]

    # calculatin statistics for each interpolation type
    for j = 1:length(param_list)
        calc_option_value_put = K -> calc_option_value(option, param_list[j], K, "Put")
        calc_option_value_call = K -> calc_option_value(option, param_list[j], K, "Call")

        IV1_raw = K -> calc_option_value_call(K)/K^2
        IV2_raw = K -> calc_option_value_put(K)/K^2

        min_K = minimum(strikes)
        max_K = maximum(strikes)
        IV_limit_i = (exp(r*T)*2/T) * (hquadrature(IV1_raw, spot, max_K)[1] + hquadrature(IV2_raw, min_K, spot)[1] - exp(-r*T)*(exp(r*T)-1-r*T))

        V_i, IV_i, D_i = calc_variation_and_jump_risk(option, param_list[j])

        to_append = DataFrame(date = [option.date], exdate = [option.exdate], fit_type = [fit_type_list[j]],
            VIX = [VIX_i], IV_limit = [IV_limit_i], IV = [IV_i], V = [V_i], D = [D_i])

        append!(variation_comp, to_append)
    end
end

CSV.write("output/variation_comp_2008_30.csv",  variation_comp)

############################################################
# Plotting the results:
# Loading actual VIX series:
# actual_vix = CSV.read("data/actual_vix_2017.csv"; datarow = 2, delim = ",")
actual_vix = CSV.read("data/actual_vix_2008.csv"; datarow = 2, delim = ",")
dates_actual_vix = actual_vix[1][unique(indexin(variation_comp[1], actual_vix[1]))]
values_actual_vix = actual_vix[6][unique(indexin(variation_comp[1], actual_vix[1]))]

cla()
clf()
fig = figure("An example", figsize=(10,8));

fit_type_list = ["rho=0, grid", "rho=0, global", "VarRho, grid", "VarRho, global"]
# fit_type_list = ["rho=0, grid", "rho=0, global"]

ax = fig[:add_subplot](1,1,1);
df_sub = variation_comp[variation_comp.fit_type .== fit_type_list[1],:]
ax[:plot](dates_actual_vix, values_actual_vix, marker = "o", alpha = 0.5, linewidth = 1, c = "m", label = "Actual VIX (Adj Close)")
ax[:plot](df_sub[:date], df_sub[:VIX], alpha = 0.5, linewidth = 2.5, c = "k", label = "Calculated VIX")

color_list = ["b", "g", "r", "c"]

for i = 1:length(fit_type_list)
    df_sub = variation_comp[variation_comp.fit_type .== fit_type_list[i],:]

    ax[:plot](df_sub[:date], 100*sqrt.(df_sub[:IV_limit]), alpha = 0.5, c = color_list[i], label = fit_type_list[i])
    ax[:plot](df_sub[:date], 100*sqrt.(df_sub[:IV]), alpha = 0.5, c = color_list[i], linestyle = "--")
end
legend()
# filepath_to_save = string("images/time_series_of_variation_measures_30_only.pdf")
filepath_to_save = string("images/TSofIV_all_2008.pdf")
PyPlot.savefig(filepath_to_save, format="pdf", bbox_inches= "tight");

############################################################
# Plotting time series of D for different parameters of SVI:

# fit_type_list = ["rho=0, grid", "rho=0, global", "VarRho, grid", "VarRho, global"]
fit_type_list = ["rho=0, grid", "rho=0, global"]

cla()
clf()
fig = figure(figsize=(10,8));
ax = fig[:add_subplot](1,1,1);

color_list = ["b", "g", "r", "c"]

for i = 1:length(fit_type_list)
    df_sub = variation_comp[variation_comp.fit_type .== fit_type_list[i],:]

    ax[:plot](df_sub[:date], 100*df_sub[:D], alpha = 0.5, c = color_list[i], label = fit_type_list[i])
end
legend()

# filepath_to_save = string("images/time_series_of_variation_measures_30_only.pdf")
filepath_to_save = string("images/TSofD_zero_rho_2008.pdf")
PyPlot.savefig(filepath_to_save, format="pdf", bbox_inches= "tight");

####################################################################
# Checking the sensitivity of IV estimates to integration region.
####################################################################

function calc_IV(option::OptionData, interp_params, low_limit, high_limit)
    spot = option.spot
    r = option.int_rate
    T = option.T

    calc_option_value_put = K -> calc_option_value(option, interp_params, K, "Put")
    calc_option_value_call = K -> calc_option_value(option, interp_params, K, "Call")

    if isequal(high_limit, Inf)
        IV1_raw = K -> calc_option_value_call(K)/K^2

        if low_limit > spot
            IV1 = t -> IV1_raw(low_limit + t/(1-t))/(1-t)^2
            IV = (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
        else
            IV1 = t -> IV1_raw(spot + t/(1-t))/(1-t)^2

            IV2_raw = K -> calc_option_value_put(K)/K^2
            IV2 = t -> IV2_raw(spot * t) * spot

            IV = (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1)[1] + hquadrature(IV2, 0, 1)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
        end
    else
        IV1 = K -> calc_option_value_call(K)/K^2
        IV2 = K -> calc_option_value_put(K)/K^2

        if low_limit > spot
            IV = (exp(r*T)*2/T) * (hquadrature(IV1, low_limit, high_limit)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
        elseif high_limit < spot
            IV = (exp(r*T)*2/T) * (hquadrature(IV2, low_limit, high_limit)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
        else
            IV = (exp(r*T)*2/T) * (hquadrature(IV1, spot, high_limit)[1] + hquadrature(IV2, low_limit, spot)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
        end
    end

    return IV
end

sensitivity_comp = DataFrame(date = [], exdate = [], fit_type = [],
    spot = [], sigmaNTM = [], IV = [], IV_in_sample = [], IV_2_to_2 = [],
    IV_5_to_2 = [], IV_5_to_5 = [])

fit_type_list = ["rho=0, grid", "rho=0, global"] #, "VarRho, grid", "VarRho, global"]

for i = 1:length(option_arr_close_30)
# for i = 1:length(option_arr_exactly_30)
    print("Option # ")
    print(i)
    print("/")
    print(length(option_arr_close_30))
    print("\n")

    option = option_arr_close_30[i]
    # option = option_arr_exactly_30[i]

    # Calculating near-the-money volatility by averaging implied vol
    # of near-the-money options
    NTM_dist = 0.05
    NTM_index = (option.strikes .<= option.spot*(1.0 + NTM_dist)) .&
                (option.strikes .>= option.spot*(1.0 - NTM_dist))
    sigma_NTM_i = mean(option.impl_vol[NTM_index])
    sigma_adj = sigma_NTM_i * sqrt(1/12)

    # Fitting different SVIs
    svi_params_1 = fit_svi_bdbg_smile_grid(option)
    svi_params_2 = fit_svi_bdbg_smile_global(option)
    param_list = [svi_params_1, svi_params_2]

    min_K = minimum(option.strikes)
    max_K = maximum(option.strikes)
    spot = option.spot

    # calculatin statistics for each interpolation type
    for j = 1:length(param_list)
        IV_i = calc_IV(option, param_list[j], 0, Inf)
        IV_in_sample_i = calc_IV(option, param_list[j], min_K, max_K)
        IV_2_to_2_i = calc_IV(option, param_list[j], maximum([0, spot*(1 - 2 * sigma_adj)]), spot*(1 + 2 * sigma_adj))
        IV_5_to_2_i = calc_IV(option, param_list[j], maximum([0, spot*(1 - 5 * sigma_adj)]), spot*(1 + 2 * sigma_adj))
        IV_5_to_5_i = calc_IV(option, param_list[j], maximum([0, spot*(1 - 5 * sigma_adj)]), spot*(1 + 5 * sigma_adj))

        to_append = DataFrame(date = [option.date], exdate = [option.exdate],
            fit_type = [fit_type_list[j]],
            spot = [spot], sigmaNTM = [sigma_NTM_i],
            IV = [IV_i], IV_in_sample = [IV_in_sample_i], IV_2_to_2 = [IV_2_to_2_i],
            IV_5_to_2 = [IV_5_to_2_i], IV_5_to_5 = [IV_5_to_5_i])

        append!(sensitivity_comp, to_append)
    end
end

CSV.write("output/sensitivity_comp_2008.csv",  sensitivity_comp)

########################################################
# Plotting the results of sensitivity comparison:
cla()
clf()
fig = figure("An example", figsize=(10,8));

ax1 = fig[:add_subplot](3,1,1);
ax2 = fig[:add_subplot](3,1,2);
ax3 = fig[:add_subplot](3,1,3);

df_sub = sensitivity_comp[sensitivity_comp.fit_type .== fit_type_list[1],:]

ax1[:plot](df_sub.date, df_sub.sigmaNTM.*sqrt(1/12), label = "Sigma NTM")
ax1[:set_title]("(Near-the-Money sigma)*sqrt(1/12)")

ax2[:plot](df_sub.date, df_sub.IV, alpha = 0.5, label = "IV")
ax2[:plot](df_sub.date, df_sub.IV_in_sample, alpha = 0.5, label = "IV in sample")
ax2[:plot](df_sub.date, df_sub.IV_2_to_2, alpha = 0.5, label = "IV for [-2sigma, 2sigma]")
ax2[:plot](df_sub.date, df_sub.IV_5_to_2, alpha = 0.5, label = "IV for [-5sigma, 2sigma]")
ax2[:plot](df_sub.date, df_sub.IV_5_to_5, alpha = 0.5, label = "IV for [-5sigma, 5sigma]")
ax2[:set_title]("SVI with rho=0, grid")

df_sub = sensitivity_comp[sensitivity_comp.fit_type .== fit_type_list[2],:]

ax3[:plot](df_sub.date, df_sub.IV, alpha = 0.5, label = "IV")
ax3[:plot](df_sub.date, df_sub.IV_in_sample, alpha = 0.5, label = "IV in sample")
ax3[:plot](df_sub.date, df_sub.IV_2_to_2, alpha = 0.5, label = "IV for [-2sigma, 2sigma]")
ax3[:plot](df_sub.date, df_sub.IV_5_to_2, alpha = 0.5, label = "IV for [-5sigma, 2sigma]")
ax3[:plot](df_sub.date, df_sub.IV_5_to_5, alpha = 0.5, label = "IV for [-5sigma, 5sigma]")
ax3[:set_title]("SVI with rho=0, global")

subplots_adjust(wspace = 0.35, hspace = 0.35);

legend()
# filepath_to_save = string("images/time_series_of_variation_measures_30_only.pdf")
filepath_to_save = string("images/sensitivity_comparison_2008.pdf")
PyPlot.savefig(filepath_to_save, format="pdf", bbox_inches= "tight");
