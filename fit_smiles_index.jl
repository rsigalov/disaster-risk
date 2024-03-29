using Distributed

@everywhere using Pkg
@everywhere Pkg.activate("DRjulia")


using CSV
using Dates

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

print("\n--- Loading Data ----\n")

index_to_append = ARGS[1]
# index_to_append = "equity_1"

# Loading data on options:
opt_data_filepath = string("data/raw_data/opt_data_", index_to_append, ".csv")
df = DataFrame(CSV.File(opt_data_filepath))
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
df_unique_N = combine(groupby(df, [:secid, :date, :exdate]), :cp_flag => length)

# If don't have at least 5 observations throw this option out since
# we need to minimize over 4 variables:
df_unique = filter(row -> row.cp_flag_length >= 5, df_unique_N)[:, [:secid,:date,:exdate]]
num_options = size(df_unique)[1]
volume_arr = zeros(num_options) # Array to store sum of volume for each maturity
open_interest_arr = zeros(num_options) # Array to store sum of open interest for each maturity
# num_options = 1000

print(string("\nHave ", num_options, " smiles in total to fit\n"))

# Loading data on index dividend yield:
div_yield_filepath = string("data/raw_data/div_yield_", index_to_append, ".csv")
div_yield = DataFrame(CSV.File(div_yield_filepath))
div_yield = sort(div_yield, [:secid, :date])

# Loading data on interest rate to interpolate cont-compounded rate:
zcb = DataFrame(CSV.File("data/raw_data/zcb_data.csv"))
zcb = sort(zcb, [:date, :days])

print("\n--- Generating array with options ----\n")
option_arr = Array{OptionData, 1}(undef, num_options)
i_option = 0

df = sort(df, [:secid, :date, :exdate, :strike_price])
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


        div_yield_sub = div_yield[(div_yield.date .== obs_date) .&
                                  (div_yield.secid .== secid), :rate]
        if size(div_yield_sub)[1] == 0
            div_yield_cur = 0
        else
            div_yield_cur = div_yield_sub[1]/100
        end

        forward = exp(-div_yield_cur*T)*spot/exp(-int_rate*T)

        strikes = subdf.strike_price./1000
        impl_vol = subdf.impl_volatility
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
                         volume = volume_arr,
                         open_inetrest = open_interest_arr,
                         m = map(x -> x.m, svi_arr),
                         sigma = map(x -> x.sigma, svi_arr),
                         a = map(x -> x.a, svi_arr),
                         b = map(x -> x.b, svi_arr),
                         obj = map(x -> x.obj, svi_arr),
                         opt_out = map(x -> x.opt_result, svi_arr))

CSV.write(string("data/raw_data/svi_params_", index_to_append, ".csv"), svi_data_out)
