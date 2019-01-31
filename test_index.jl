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
include("funcs.jl")

# df = CSV.read("data/opt_data.csv"; datarow = 2, delim = ",")
df = CSV.read("data/opt_data_2.csv"; datarow = 2, delim = ",")
df_unique = unique(df[:, [:secid,:date,:exdate]])

# Loading data on interest rate to interpolate cont-compounded rate:
zcb = CSV.read("data/zcb_cont_comp_rate.csv"; datarow = 2, delim = ",")
zcb = sort(zcb, [:date, :days])

spx_div_yield = CSV.read("data/spx_dividend_yield.csv"; datarow = 2, delim = ",")
spx_div_yield = sort(spx_div_yield, [:secid, :date])

max_options = 100
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
    T = opt_days_maturity/365

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

function BS_call_price(S0, q, r, K, sigma, T)
    d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    p1 = exp(-q*T) * S0 * cdf.(Normal(), d1)
    p2 = exp(-r*T) * K * cdf.(Normal(), d2)

    return p1 - p2
end

# Calculating BS put price:
function BS_put_price(S0, q, r, K, sigma, T)
    d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    p1 = cdf.(Normal(), -d2) * K * exp(-r*T)
    p2 = cdf.(Normal(), -d1) * S0 * exp(-q*T)

    return p1 - p2
end

################################################################
# Solve for Black-Scholes parameters using numerical solver. Have
# two unknowns: dividend yield q and interest rate r.

# 1. Getting prices and strikes of specified options:
i_option = 1
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

####################################################
# Semi-manually calculating BS prices:

obs_date = df_unique[:date][1]
exp_date = df_unique[:exdate][1]
T = Dates.value(exp_date - obs_date)/365

df_sub = df[(df.date .== obs_date) .& (df.exdate .== exp_date), :]
df_sub = sort(df_sub, :strike_price)
mid_price = df_sub[:mid_price][120]
impl_vol = df_sub[:impl_volatility][120]
spot = df_sub[:under_price][120]
strike = df_sub[:strike_price][120]/1000

zcb_sub = zcb[zcb.date .== obs_date, :]
r1 = zcb_sub[:rate][3]
days1 = zcb_sub[:days][3]
r2 = zcb_sub[:rate][4]
days2 = zcb_sub[:days][4]

r = r1 + (r2 - r1)*(T*365 - days1)/(days2 - days1)
r = r/100

df_yield_sub = spx_div_yield[spx_div_yield.date .== obs_date, :]
q = df_yield_sub[:rate][1]/100

d1 = (log(spot/strike) + T*(r - q + 0.5*impl_vol)^2)/(impl_vol*sqrt(T))
d2 = d1 - impl_vol*sqrt(T)

put_price = exp(-r*T)*strike*cdf.(Normal(), -d2) - exp(-q*T)*spot*cdf.(Normal(), -d1)





function f(x)
    return cos(x)
end

hquadrature(f, -1, 100)
