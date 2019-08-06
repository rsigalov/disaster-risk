using Distributed

print("\nNumber of processors ")
print(nprocs())
print("\n")

print("\n ---- Loading libraries ----\n")

# @everywhere using DataFrames # self explanatory
using DataFrames
@everywhere using NLopt # Package to perform numerical optiimization
@everywhere using LinearAlgebra # Package with some useful functions
@everywhere using Distributions # Package for normal CDF
@everywhere using HCubature # Package to numerically integrate
@everywhere using ForwardDiff # Package to numerically differentiate
@everywhere using Dierckx # Package for interpolation
@everywhere include("funcs.jl")

using CSV
using Dates

print("\n ---- Loading Data ----\n")

################################################################################
################################################################################
### Uncomment the following part if want to estimate implied volatilities
# Loading data on options, interest rates and dividend yields:
# df = CSV.read("data/raw_data/opt_data_spx_all_CME.csv"; datarow = 2, delim = ",")
#
# # Calculating forward rate for all options:
# df[:forward] = exp.(-df[:div_yield] .* df[:T]) .* df[:spot]./exp.(-df[:int_rate].*df[:T])
#
# # For each individual option calculating implied volatility:
# price_arr = df[:price]
# F_exprt_arr = df[:forward] .* exp.(-df[:int_rate].*df[:T])
# int_rate_arr = df[:int_rate]
# strike_arr = df[:strike_price]
# T_arr = df[:T]
# option_type_arr = df[:cp_flag]
#
# print("\n ---- First Pass for Implied Vol ----\n")
#
# @time tmp = pmap(calc_implied_vol, price_arr[1:2], F_exprt_arr[1:2], zeros(2),
#     int_rate_arr[1:2], strike_arr[1:2], T_arr[1:2], option_type_arr[1:2])
#
# print("\n ---- Second Pass for Implied Vol ----\n")
#
# @time impl_vol = pmap(calc_implied_vol, price_arr, F_exprt_arr, zeros(length(price_arr)),
#     int_rate_arr, strike_arr, T_arr, option_type_arr)
#
# df[:impl_volatility] = impl_vol
#
# print("\n ---- Saving Implied Vol ----\n")
#
# CSV.write(string("data/raw_data/opt_data_spx_all_CME.csv"), df)
################################################################################
################################################################################

################################################################
# Now working with data that has implied volatility
################################################################
# Loading the file back again:
df = CSV.read("data/raw_data/opt_data_spx_all_CME.csv"; datarow = 2, delim = ",")
df = df[df.date .!== df.exdate, :]


# Calculating number of options per secid, observation date and expiration date
df_unique_N = by(df, [:date, :exdate], number = :cp_flag => length)

# If don't have at least 5 observations throw this option out since
# we need to minimize over 4 variables:
df_unique = df_unique_N[df_unique_N[:number] .>= 5, :][:, [:date,:exdate]]
num_options = size(df_unique)[1] # number of options to fit smiles in
volume_arr = zeros(num_options) # Array to store sum of volume for each maturity
open_interest_arr = zeros(num_options) # Array to store sum of open interest for each maturity

print(string("\nHave ", num_options, " smiles in total to fit\n"))

print("\n--- Generating array with options ----\n")
option_arr = Array{OptionData, 1}(undef, num_options)
i_option = 0

df = sort(df, cols = [:date, :exdate, :strike_price])
for subdf in groupby(df, [:date, :exdate])
    if i_option % 2500 == 0
        print(string("Preparing option smile ", i_option, " out of ", num_options, "\n"))
    end
    if size(subdf)[1] >= 5 # include only smiles with at least 5 observations:
        obs_date = subdf.date[1]
        exp_date = subdf.exdate[1]
        spot = subdf.spot[1]
        opt_days_maturity = Dates.value(exp_date - obs_date)
        T = (opt_days_maturity - 1)/365
        int_rate = subdf.int_rate[1]
        div_yield_cur = subdf.div_yield[1]
        forward = subdf.forward[1]

        strikes = subdf.strike_price
        impl_vol = subdf.impl_volatility
        if (length(strikes) >= 5) & (forward > 0)
            global i_option += 1
            option_arr[i_option] = OptionData(108105, obs_date, exp_date, spot, strikes,
                                              impl_vol, T, int_rate, forward)
            # volume_arr[i_option] = sum(subdf.volume)
            open_interest_arr[i_option] = sum(subdf.open_interest)
        end
    end
end

option_arr = option_arr[1:i_option]
volume_arr = volume_arr[1:i_option]
open_interest_arr = open_interest_arr[1:i_option]
num_options = length(option_arr) # Updating number of smiles to count only those
                                 # that have at least 5 options available after
                                 # additional present value filter

print(string("\n--- ", num_options, " left after processing ---\n"))
print("\n--- Doing stuff ---")
print("\n--- Fitting SVI Volatility Smile ---\n")
print("\n--- First Pass ---\n")
@time tmp = pmap(fit_svi_zero_rho_global, option_arr[1:2])
print("\n--- Second Pass ---\n")
@time svi_arr = pmap(fit_svi_zero_rho_global, option_arr)

print("\n--- Saving required data to estimate parameters ---\n")

# Calculating near the money sigma:
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

# Saving SVI parameters into a dataset:
svi_data_out = DataFrame(secid = map(x -> x.secid, option_arr),
                         obs_date = map(x -> x.date, option_arr),
                         exp_date = map(x -> x.exdate, option_arr),
                         spot = map(x -> x.spot, option_arr),
                         T = map(x -> x.T, option_arr),
                         r = map(x -> x.int_rate, option_arr),
                         F = map(x -> x.forward, option_arr),
                         sigma_NTM = map(x -> calc_NTM_sigma(x), option_arr),
                         min_K = map(x -> minimum(x.strikes), option_arr),
                         max_K = map(x -> maximum(x.strikes), option_arr),
                         # volume = volume_arr,
                         open_inetrest = open_interest_arr,
                         m = map(x -> x.m, svi_arr),
                         sigma = map(x -> x.sigma, svi_arr),
                         a = map(x -> x.a, svi_arr),
                         b = map(x -> x.b, svi_arr),
                         obj = map(x -> x.obj, svi_arr),
                         opt_out = map(x -> x.opt_result, svi_arr))

CSV.write(string("data/raw_data/svi_params_spx_all_CME.csv"), svi_data_out)
