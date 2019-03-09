using DataFrames
using NLopt # Package to perform numerical optiimization
using LinearAlgebra # Package with some useful functions
using Distributions # Package for normal CDF
using HCubature # Package to numerically integrate
using ForwardDiff # Package to numerically differentiate
using Dierckx # Package for interpolation
include("funcs.jl")

using CSV
using Dates


using Plots
using PyPlot





# Loading data on options:
df = CSV.read("aapl_opt_example_data.csv"; datarow = 2, delim = ",")
df_size = size(df)[1]
print(string("\n--- Total size of dataframe is ", df_size, " rows ---\n"))

# Calculating number of options per secid, observation date and expiration date
df_unique_N = by(df, [:secid, :date, :exdate], number = :cp_flag => length)

# If don't have at least 5 observations throw this option out since
# we need to minimize over 4 variables:
df_unique = df_unique_N[df_unique_N[:number] .>= 5, :][:, [:secid,:date,:exdate]]
num_options = size(df_unique)[1]

# Loading data on dividend distribution:
dist_hist = CSV.read("aapl_dist_example_data.csv"; datarow = 2, delim = ",")

# Loading data on interest rate to interpolate cont-compounded rate:
zcb = CSV.read("data/zcb_data.csv"; datarow = 2, delim = ",")
zcb = sort(zcb, [:date, :days])

print("\n--- Generating array with options ----\n")
option_arr = Array{OptionData, 1}(undef, num_options)
i_option = 0

df = sort(df, cols = [:secid, :date, :exdate, :strike_price])
for subdf in groupby(df, [:secid, :date, :exdate])
    if i_option % 2500 == 0
        print(string("Preparing option smile ", i_option, " out of ", num_options, "\n"))
    end
    if size(subdf)[1] >= 5 # include only smiles with at least 5 observations:
        obs_date = subdf.date[1]
        exp_date = subdf.exdate[1]
        secid = subdf.secid[1]
        # print(string(obs_date," ",exp_date, " ", "\n"))

        spot = subdf.under_price[1]
        opt_days_maturity = Dates.value(exp_date - obs_date)
        T = (opt_days_maturity - 1)/365

        subzcb = zcb[zcb.date .== obs_date,:]
        if size(subzcb)[1] == 0
            subzcb = zcb[zcb.date .<= obs_date,:]
            prev_obs_date = subzcb.date[end]
            subzcb = zcb[zcb.date .== prev_obs_date,:]
        end
        x = subzcb.days
        y = subzcb.rate
        interp_rate = Spline1D(x, y, k = 1) # creating linear interpolation object
                                            # that we can use later as well

        int_rate = interp_rate(opt_days_maturity - 1)/100

        index_before = (dist_hist.secid .== secid) .& (dist_hist.ex_date .<= exp_date) .& (dist_hist.ex_date .>= obs_date)
        if count(index_before) == 0
            dist_pvs = [0.0]
        else
            dist_days = Dates.value.(dist_hist[index_before, :].ex_date .- obs_date) .- 1
            dist_amounts = dist_hist[index_before, :].amount

            dist_rates = map(days -> interp_rate(days), dist_days)./100

            dist_pvs = exp.(-dist_rates .* dist_days/365) .* dist_amounts
        end

        forward = (spot - sum(dist_pvs))/exp(-int_rate .* T)

        ############################################################
        ### Additional filter related to present value of strike and dividends:
        ### Other filters are implemented in SQL query directly
        # For call options we should have C >= max{0, spot - PV(K) - PV(dividends)}
        # For Put options we should have P >= max{0, PV(K) + PV(dividends) - spot}
        # If options for certain strikes violate these conditions we should remove
        # them from the set of strikes
        strikes_put = subdf[subdf.cp_flag .== "P",:strike_price]./1000
        strikes_call = subdf[subdf.cp_flag .== "C", :strike_price]./1000
        call_min = max.(0, spot .- strikes_call .* exp(-int_rate * T) .- sum(dist_pvs))
        put_min = max.(0, strikes_put .* exp(-int_rate*T) .+ sum(dist_pvs) .- spot)

        df_filter = subdf[subdf.mid_price .>= [put_min; call_min],:]
        strikes = df_filter.strike_price./1000
        impl_vol = df_filter.impl_volatility
        if (length(strikes) >= 5) & (forward > 0)
            global i_option += 1
            option_arr[i_option] = OptionData(secid, obs_date, exp_date, spot, strikes,
                                              impl_vol, T, int_rate, forward)
        end
    end
end

option_arr = option_arr[1:i_option]
num_options = length(option_arr) # Updating number of smiles to count only those
                                 # that have at least 5 options available after
                                 # additional present value filter

# Fittin options:
svi_arr = map(fit_svi_zero_rho_global, option_arr)

# Plotting the fit:
for i_option = 1:num_options
    clf()
    fig = figure("An example", figsize=(6,4));
    ax = fig[:add_subplot](1,1,1);
    plot_vol_smile(option_arr[i_option], svi_arr[i_option], "", ax);
    title_text = string(option_arr[i_option].secid,", from ", option_arr[i_option].date, " to ", option_arr[i_option].exdate);
    suptitle(title_text);
    filepath_to_save = string("images/ind_option_test/NKE_example_fit_", i_option, ".pdf")
    PyPlot.savefig(filepath_to_save, format="pdf", bbox_inches= "tight");
end

# Estimating parameters:
function calc_NTM_sigma(option::OptionData)
    NTM_dist = 0.05
    NTM_index = (option.strikes .<= option.spot*(1.0 + NTM_dist)) .&
                (option.strikes .>= option.spot*(1.0 - NTM_dist))
    if sum(NTM_index) == 0
        if minimum(option.strikes) > option.spot
            sigma_NTM = option.impl_vol[1]
        elseif maximum(option.strikes) < option.spot
            sigma_NTM = option.impl_vol[end]
        else
            sigma_NTM = option.impl_vol[option.strikes .<= option.spot][end]
        end
    else
        sigma_NTM = mean(option.impl_vol[NTM_index]) * sqrt(option.T)
    end
    return sigma_NTM
end

spot_arr = map(x -> x.spot, option_arr)
r_arr = map(x -> x.int_rate, option_arr)
F_arr = map(x -> x.forward, option_arr)
T_arr = map(x -> x.T, option_arr)
sigma_NTM_arr = map(x -> calc_NTM_sigma(x), option_arr)
min_K_arr = map(x -> minimum(x.strikes), option_arr)
max_K_arr = map(x -> maximum(x.strikes), option_arr)
ests = map(estimate_parameters, spot_arr, r_arr,
           F_arr, T_arr, sigma_NTM_arr,
           min_K_arr, max_K_arr, svi_arr)


#########################################################
# Testing which part of the integral actually diverges:
#########################################################
option = option_arr[3]
interp_params = svi_arr[3]

spot = option.spot
T = option.T
r = option.int_rate
F = option.forward

high_limit = Inf
low_limit = 0

# In this case we are dealing with both integrals with calls and puts
# First, dealing with integrated variation:
calc_option_value_put = K -> calc_option_value(spot, r, F, T, interp_params, K, "Put")
calc_option_value_call = K -> calc_option_value(spot, r, F, T, interp_params, K, "Call")

IV1_raw = K -> calc_option_value_call(K)/K^2
IV1 = t -> IV1_raw(spot + t/(1-t))/(1-t)^2

IV2_raw = K -> calc_option_value_put(K)/K^2
IV2 = t -> IV2_raw(spot * t) * spot

integrated_variation =
    (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1, maxevals = 100000)[1] +
    hquadrature(IV2, 0, 1, maxevals = 100000)[1] -
    exp(-r*T)*(exp(r*T)-1-r*T))

# These integrals contain call(strike) function:
V1_raw = K -> 2 * (1 - log(K/spot)) * calc_option_value_call(K)/K^2
W1_raw = K -> (6 * log(K/spot) - 3 * (log(K/spot))^2) * calc_option_value_call(K)/K^2
X1_raw = K -> (12 * log(K/spot)^2 - 4 * log(K/spot)^3) * calc_option_value_call(K)/K^2

# These integrals contain put(strike) function
V2_raw = K -> 2 * (1 + log(spot/K)) * calc_option_value_put(K)/K^2
W2_raw = K -> (6 * log(spot/K) + 3 * log(spot/K)^2) * calc_option_value_put(K)/K^2
X2_raw = K -> (12 * log(spot/K)^2 + 4 * log(spot/K)^3) * calc_option_value_put(K)/K^2

# Modifying integrands to account for infinite upper integration limit
V1 = t -> V1_raw(spot + t/(1-t))/(1-t)^2
W1 = t -> W1_raw(spot + t/(1-t))/(1-t)^2
X1 = t -> X1_raw(spot + t/(1-t))/(1-t)^2

V = hquadrature(V1, 0, 1, maxevals = 100000)[1] + hquadrature(V2_raw, low_limit, spot, maxevals = 100000)[1]
W = hquadrature(W1, 0, 1, maxevals = 100000)[1] + hquadrature(W2_raw, low_limit, spot, maxevals = 100000)[1]
X = hquadrature(X1, 0, 1, maxevals = 100000)[1] + hquadrature(X2_raw, low_limit, spot, maxevals = 100000)[1]

mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)

K_range = LinRange(0.01, 300, 3000)

Plots.plot(K_range, IV2_raw.(K_range))

Plots.plot(K_range, calc_option_value_put.(K_range))





################################################
# Interpolating with BSplines:
################################################

option = option_arr[3]
log_moneyness = log.(option.strikes/option.spot)

spl = Spline1D(log_moneyness, option.impl_vol; w = ones(length(log_moneyness)),
               k=3, bc="nearest", s=0.25)

clf()
fig = figure("An example", figsize=(6,4));
ax = fig[:add_subplot](1,1,1);
plot_vol_smile(option, spl, "", ax);
title_text = string("AAPL, from ", option_arr[i_option].date, " to ", option_arr[i_option].exdate);
suptitle(title_text);
filepath_to_save = string("aapl_example_fit_spline.pdf")
PyPlot.savefig(filepath_to_save, format="pdf", bbox_inches= "tight");









# Adding some artificial option to facilitate smooth transition:
# option = option_arr[5]
# strikes = option.strikes
# impl_vol = option.impl_vol
# log_moneyness = log.(option.strikes/option.spot)
#
# log_moneyness = [LinRange(-10,log_moneyness[1],1000); log_moneyness; LinRange(log_moneyness[end],10,1000)]
# impl_vol = [ones(1000)*impl_vol[1]; impl_vol; ones(1000)*impl_vol[end]]
#
# spl = Spline1D(log_moneyness, impl_vol.^2;
#                w = [0.5 * ones(1000);ones(length(strikes)); 0.5 * ones(1000)],
#                k=3, bc="nearest", s = 0.01)
#
# clf()
# fig = figure("An example", figsize=(6,4));
# ax = fig[:add_subplot](1,1,1);
# plot_vol_smile(option, spl, "", ax);
# title_text = string("AAPL, from ", option_arr[i_option].date, " to ", option_arr[i_option].exdate);
# suptitle(title_text);
# filepath_to_save = string("aapl_example_fit_spline_extended.pdf")
# PyPlot.savefig(filepath_to_save, format="pdf", bbox_inches= "tight");


####################################################
# Testing clamped SVI parametrization of vol smile
####################################################
