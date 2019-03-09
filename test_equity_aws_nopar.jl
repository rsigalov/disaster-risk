
print(ARGS)
print("\n")
print(nprocs())

@everywhere using DataFrames
@everywhere using NLopt
@everywhere include("funcs_aws.jl")

@everywhere using Distributions
@everywhere using HCubature
@everywhere using ForwardDiff

using CSV
using Dates

# Loading data on options:
df = CSV.read("option_data/test_opt_data.csv"; datarow = 2, delim = ",", nullable = false)

# Calculating number of options per secid, observation date and expiration date
df_unique_N = by(df, [:secid, :date, :exdate], d -> length(d[:secid]))

# If don't have at least 5 observations throw this option out since
# we need to minimize over 4 variables:
df_unique = df_unique_N[df_unique_N[:x1] .>= 5, :][:, [:secid,:date,:exdate]]
num_options = size(df_unique)[1]

# Loading data on interest rate to interpolate cont-compounded rate:
zcb = CSV.read("option_data/test_zcb_data.csv"; datarow = 2, delim = ",", nullable = false)
zcb = sort(zcb, cols = (:date, :days))

# Loading data on dividend distribution:
dist_hist = CSV.read("option_data/test_dist_data.csv"; datarow = 2, delim = ",", nullable = false)

option_arr = Array{OptionData, 1}(num_options)

print("\n ---- Generating array with options ----\n")

for i_option = 1:num_options
    # print("Option # ")
    # print(i_option)
    # print("\n")
    obs_date = df_unique[:date][i_option]
    exp_date = df_unique[:exdate][i_option]
    secid = df_unique[:secid][i_option]

    df_sub = df[(df[:date] .== obs_date) .& (df[:exdate] .== exp_date) .& (df[:secid] .== secid), :]
    df_sub = sort(df_sub, cols = :strike_price)

    # strikes = df_sub[:strike_price]./1000
    # impl_vol = df_sub[:impl_volatility]
    spot = df_sub[:under_price][1]
    opt_days_maturity = Dates.value(exp_date-obs_date)
    T = (opt_days_maturity-1)/365 # not sure what to divide with

    ############################################################
    # Using data on ZCBs to inetrpolate risk-free rate at the maturity
    # of the option
    int_rate = interpolate_int_rate(obs_date, exp_date, zcb)

    ############################################################
    # Using data on distributions to calculate their present value
    index_before = (dist_hist[:secid] .== secid) .& (dist_hist[:ex_date] .<= exp_date) .& (dist_hist[:ex_date] .>= obs_date)
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

    # Calculating forward using present value of distributions and int. rate
    forward = (spot - sum(dist_pvs))/exp(-int_rate .* T)

    ############################################################
    ### Additional filter related to present value of strike and dividends:
    ### Other filters are implemented in SQL query directly
    # For call options we should have C >= max{0, spot - PV(K) - PV(dividends)}
    # For Put options we should have P >= max{0, PV(K) + PV(dividends) - spot}
    # If options for certain strikes violate these conditions we should remove
    # them from the set of strikes
    strikes_put = df_sub[df_sub[:cp_flag] .== "P", :strike_price]./1000
    strikes_call = df_sub[df_sub[:cp_flag] .== "C", :strike_price]./1000
    call_min = max.(0, spot .- strikes_call .* exp(-int_rate * T) .- sum(dist_pvs))
    put_min = max.(0, strikes_put .* exp(-int_rate*T) .+ sum(dist_pvs) .- spot)

    df_sub[:opt_min] = [put_min; call_min]
    df_filter = df_sub[df_sub[:mid_price] .>= df_sub[:opt_min],:]
    strikes = df_filter[:strike_price]./1000
    impl_vol = df_filter[:impl_volatility]

    ############################################################
    # Writing everything into struct:
    option_arr[i_option] = OptionData(secid, obs_date, exp_date, spot, strikes,
                                      impl_vol, T, int_rate, forward)
end

@time tmp = 2+2;
print("\n---- Fitting SVI volatility smile ----\n")
print("\n---- First pass ----\n")
@time svi_arr = pmap(fit_svi_zero_rho_global, option_arr[1:2])
print("\n---- Second pass ----\n")
@time svi_arr = pmap(fit_svi_zero_rho_global, option_arr)
print("\n---- Calculating statistics ----\n")
print("\n---- First pass ----\n")
@time ests = pmap(estimate_parameters, option_arr[1:2], svi_arr[1:2])
print("\n---- Second pass ----\n")
@time ests = pmap(estimate_parameters, option_arr, svi_arr)

print("\nDone\n\n")
