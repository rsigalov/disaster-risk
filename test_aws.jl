print("Number of processors: ")
print(nprocs())
print("\n")

@everywhere using DataFrames
@everywhere using NLopt
@everywhere include("funcs_aws.jl")

using CSV
using Dates

df = CSV.read("option_data/opt_data.csv"; datarow = 2, delim = ",")
# workaround to covert nullable arrays to normal arrays
for j = 1:length(names(df))
    df[names(df)[j]]=convert(Array, df[names(df)[j]])
end
df_unique = unique(df[:, [:secid,:date,:exdate]])
df_unique = sort(df_unique, cols = (:secid, :date, :exdate))

zcb = CSV.read("option_data/zcb_data.csv"; datarow = 2, delim = ",")
for j = 1:length(names(zcb))
    zcb[names(zcb)[j]]=convert(Array, zcb[names(zcb)[j]])
end
zcb = sort(zcb, cols = (:date, :days))

spx_div_yield = CSV.read("option_data/spx_dividend_yield.csv"; datarow = 2, delim = ",")
for j = 1:length(names(spx_div_yield))
    spx_div_yield[names(spx_div_yield)[j]]=convert(Array, spx_div_yield[names(spx_div_yield)[j]])
end
spx_div_yield = sort(spx_div_yield, cols = :date)

max_options = size(df_unique)[1]

option_arr = Array{OptionData, 1}(max_options)

for i_option = 1:max_options
    print(i_option)
    print("\n")
    obs_date = df_unique[:date][i_option]
    exp_date = df_unique[:exdate][i_option]
    secid = df_unique[:secid][i_option]

    df_sub = df[(df[:date] .== obs_date) .& (df[:exdate] .== exp_date) .& (df[:secid] .== secid), :]

    df_sub = sort(df_sub, cols = :strike_price)
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
    div_yield_cur = spx_div_yield[(spx_div_yield[:date] .== obs_date) .&
                                  (spx_div_yield[:secid] .== secid), :rate][1]/100

    forward = exp(-div_yield_cur*T)*spot/exp(-int_rate*T)

    option_arr[i_option] = OptionData(secid, obs_date, exp_date, spot, strikes,
                                      impl_vol, T, int_rate, forward)
end

@time tmp = pmap(fit_svi_zero_rho_grid, option_arr[1:2]);
@time svi_zero_rho_grid_params_arr = pmap(fit_svi_zero_rho_grid, option_arr);

@time tmp = pmap(fit_svi_zero_rho_global, option_arr[1:2]);
@time svi_zero_rho_global_params_arr = pmap(fit_svi_zero_rho_global, option_arr);
