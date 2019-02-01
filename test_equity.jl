using DataFrames
using CSV
using Dates
using NLopt
using Plots
using PyPlot
using SparseArrays
using LinearAlgebra
using Distributions
include("funcs.jl")

# Loading data on options:
df = CSV.read("data/opt_data_aapl.csv"; datarow = 2, delim = ",")
df_unique = unique(df[:, [:secid,:date,:exdate]])
num_options = size(df_unique)[1]

# Loading data on interest rate to interpolate cont-compounded rate:
zcb = CSV.read("data/zcb_cont_comp_rate.csv"; datarow = 2, delim = ",")
zcb = sort(zcb, [:date, :days])

# Loading data on dividend distribution:
dist_hist = CSV.read("data/dist_hist_aapl.csv"; datarow = 2, delim = ",")

option_arr = Array{OptionData, 1}(undef, num_options)

for i_option = 1:num_options
    obs_date = df_unique[:date][i_option]
    exp_date = df_unique[:exdate][i_option]
    secid = df_unique[:secid][i_option]

    df_sub = df[(df.date .== obs_date) .& (df.exdate .== exp_date) .& (df.secid .== secid), :]
    df_sub = sort!(df_sub, :strike_price)

    df_sub = unique(df_sub)

    strikes = df_sub[:strike_price]./1000
    impl_vol = df_sub[:impl_volatility]
    spot = df_sub[:under_price][1]
    opt_days_maturity = Dates.value(exp_date-obs_date)
    T = opt_days_maturity/365 # not sure what to divide with

    ############################################################
    # Using data on ZCBs to inetrpolate risk-free rate:
    int_rate = interpolate_int_rate(obs_date, exp_date, zcb)

    ############################################################
    # Using data on distributions to calculate their present value
    # print(i_option)
    # print("\n")
    index_before = (dist_hist.ex_date .<= exp_date) .& (dist_hist.ex_date .>= obs_date)
    if count(index_before) == 0
        dist_pvs = [0.0]
    else
        dist_dates = dist_hist[index_before, :][:ex_date]
        dist_amounts = dist_hist[index_before, :][:amount]

        function interpolate_list_int_rate(date_val)
            return interpolate_int_rate(obs_date, date_val, zcb)
        end

        dist_rates = map(interpolate_list_int_rate, dist_dates)
        dist_time_discount = Dates.value.(dist_dates .- obs_date)

        dist_pvs = exp.(-dist_rates .* dist_time_discount/365) .* dist_amounts
    end

    int_rate = interpolate_int_rate(obs_date, exp_date, zcb)

    forward = (spot - sum(dist_pvs))/exp(-int_rate .* opt_days_maturity/365)

    ############################################################
    # Writing everything into struct:
    option_arr[i_option] = OptionData(secid, obs_date, exp_date, spot, strikes,
                                      impl_vol, T, int_rate, forward)
end

################################################
# Calculating Black-Scholes Price
# function to calculate BS price for an asset with
# continuously compounded dividend at rate q. Can be
# accomodated to calculate price of option for an
# asset with discrete know ndividends
function BS_call_price(S0, q, r, K, sigma, T)
    d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    p1 = exp(-q*T) * S0 * cdf.(Normal(), d1)
    p2 = exp(-r*T) * K * cdf.(Normal(), d2)

    return p1 - p2
end

function BS_put_price(S0, q, r, K, sigma, T)
    d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    p1 = cdf.(Normal(), -d2) * K * exp(-r*T)
    p2 = cdf.(Normal(), -d1) * S0 * exp(-q*T)

    return p1 - p2
end

############################################################
# Comparing option prices calculated using given implied
# volatility and with mid-prices. Given the description
# on IvyDB they should match

for i_option = 1:100
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
    fig = figure("An example", figsize=(4,4));
    ax = fig[:add_subplot](1,1,1);

    ax[:scatter]([strikes_puts; strikes_calls], [prices_puts; prices_calls],
                 alpha = 0.25, c = "b", label = "Actual Prices")
    ax[:scatter]([strikes_puts; strikes_calls], [calc_prices_puts; calc_prices_calls],
                 alpha = 0.25, c = "r", label = "Calculated BS Prices")

    ax[:set_title]("Title")
    ax[:set_xlabel]("Strike")
    ax[:set_ylabel]("Price")
    legend()
    filename = string("images/compare_actual_and_calculated_prices/mid_price_BS_equity_", i_option,".pdf")
    PyPlot.savefig(filename, format = "pdf", bbox_inches = "tight");
end







############################################################

function BS_put_price_option(option::OptionData)
    BS_put_price(option.forward, 0, option.int_rate,
                 option.strikes[1], option.impl_vol[1], option.T)
end

map(BS_put_price_option, option_arr)

obs_date = df_unique[:date][i_option]
exp_date = df_unique[:exdate][i_option]
secid = df_unique[:secid][i_option]

df_sub = df[(df.date .== obs_date) .& (df.exdate .== exp_date) .& (df.secid .== secid), :]
df_sub = sort(df_sub, :strike_price)

###############################################################################
# Tests
###############################################################################
i_option = 200

obs_date = df_unique[:date][i_option]
exp_date = df_unique[:exdate][i_option]
secid = df_unique[:secid][i_option]

df_sub = df[(df.date .== obs_date) .& (df.exdate .== exp_date) .& (df.secid .== secid), :]
df_sub = sort!(df_sub, :strike_price)

df_sub = unique(df_sub)

strikes = df_sub[:strike_price]./1000
impl_vol = df_sub[:impl_volatility]
spot = df_sub[:under_price][1]
opt_days_maturity = Dates.value(exp_date-obs_date)
T = (opt_days_maturity-1)/365 # It seems that you need to subtract 1 day
                              # because the settlement is before the end
                              # of the day

############################################################
# Using data on ZCBs to inetrpolate risk-free rate:
int_rate = interpolate_int_rate(obs_date, exp_date, zcb)

############################################################
# Using data on distributions to calculate their present value
print(i_option)
print("\n")
index_before = (dist_hist.ex_date .<= exp_date) .& (dist_hist.ex_date .>= obs_date)
if count(index_before) == 0
    dist_pvs = [0.0]
else
    dist_dates = dist_hist[index_before, :][:ex_date]
    dist_amounts = dist_hist[index_before, :][:amount]

    function interpolate_list_int_rate(date_val)
        return interpolate_int_rate(obs_date, date_val, zcb)
    end

    dist_rates = map(interpolate_list_int_rate, dist_dates)
    dist_time_discount = Dates.value.(dist_dates .- obs_date)

    dist_pvs = exp.(-dist_rates .* dist_time_discount/365) .* dist_amounts
end

int_rate = interpolate_int_rate(obs_date, exp_date, zcb)

forward = (1/exp(-int_rate .* opt_days_maturity/365)) * (spot - sum(dist_pvs))
