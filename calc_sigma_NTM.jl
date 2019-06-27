print("\n ---- Loading libraries ----\n")

# @everywhere using DataFrames # self explanatory
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

print("\n--- Loading Data ----\n")

index_to_append = ARGS[1]
# index_to_append = "equity_1"

# Loading data on options:
opt_data_filepath = string("data/raw_data/opt_data_", index_to_append, ".csv")
df = CSV.read(opt_data_filepath; datarow = 2, delim = ",")
df_size = size(df)[1]
print(string("\n--- Total size of dataframe is ", df_size, " rows ---\n"))

if length(ARGS) > 1
    df_limit = parse(Int,ARGS[2])
    if df_limit > df_size
        df_limit = df_size
    end
    print(string("\n--- Limiting estimation to ", df_limit, " rows ---\n"))
else
    df_limit = df_size
end


# Calculating number of options per secid, observation date and expiration date
df_unique_N = by(df, [:secid, :date, :exdate], number = :cp_flag => length)

# If don't have at least 5 observations throw this option out since
# we need to minimize over 4 variables:
df_unique = df_unique_N[df_unique_N[:number] .>= 5, :][:, [:secid,:date,:exdate]]
num_options = size(df_unique)[1]
# num_options = 1000

print(string("\nHave ", num_options, " smiles in total to fit\n"))

# Loading data on dividend distribution:
dist_data_filepath = string("data/raw_data/dist_data_", index_to_append, ".csv")
dist_hist = CSV.read(dist_data_filepath; datarow = 2, delim = ",")

# Loading data on interest rate to interpolate cont-compounded rate:
zcb = CSV.read("data/raw_data/zcb_data.csv"; datarow = 2, delim = ",")
zcb = sort(zcb, [:date, :days])

print("\n--- Generating array with options ----\n")
option_arr = Array{OptionData, 1}(undef, num_options)
volume_arr = zeros(num_options) # Array to store sum of volume for each maturity
open_interest_arr = zeros(num_options) # Array to store sum of open interest for each maturity
i_option = 0

df = sort(df, cols = [:secid, :date, :exdate, :strike_price])
for subdf in groupby(df[1:df_limit, :], [:secid, :date, :exdate])
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
            volume_arr[i_option] = sum(subdf.volume)
            open_interest_arr[i_option] = sum(subdf.open_interest)
        end
    end
end

option_arr = option_arr[1:i_option]
num_options = length(option_arr) # Updating number of smiles to count only those
                                 # that have at least 5 options available after
                                 # additional present value filter

# Calculating near the money sigma, annualizing it by multiplying with SQRT(T)
# of a particular option
function calc_NTM_sigma(option::OptionData)
    NTM_dist = 0.05
    NTM_index = (option.strikes .<= option.spot*(1.0 + NTM_dist)) .&
             (option.strikes .>= option.spot*(1.0 - NTM_dist))
    if sum(NTM_index) == 0
     if minimum(option.strikes) > option.spot
         sigma_NTM = option.impl_vol[1] * sqrt(option.T)
     elseif maximum(option.strikes) < option.spot
         sigma_NTM = option.impl_vol[end] * sqrt(option.T)
     else
         sigma_NTM = option.impl_vol[option.strikes .<= option.spot][end] * sqrt(option.T)
     end
    else
     sigma_NTM = mean(option.impl_vol[NTM_index]) * sqrt(option.T)
    end
    return sigma_NTM
end

# pre-comliping the calc_NTM_sigma function:
tmp = calc_NTM_sigma(option_arr[1])

# Saving SVI parameters into a dataset:
svi_data_out = DataFrame(
    secid = map(x -> x.secid, option_arr),
    obs_date = map(x -> x.date, option_arr),
    exp_date = map(x -> x.exdate, option_arr),
    sigma_NTM = map(x -> calc_NTM_sigma(x), option_arr))

CSV.write(string("data/sigma_NTM/sigma_NTM_", index_to_append, ".csv"), svi_data_out)
