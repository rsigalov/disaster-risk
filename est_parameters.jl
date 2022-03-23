using Distributed

@everywhere using Pkg
@everywhere Pkg.activate("DRjulia")

using Distributed

print("\nNumber of processors ")
print(nprocs())
print("\n")

print("\n ---- Loading libraries ----\n")

@everywhere using DataFrames # self explanatory
@everywhere using LinearAlgebra # Package with some useful functions
@everywhere using Distributions # Package for normal CDF
@everywhere using HCubature # Package to numerically integrate
@everywhere using ForwardDiff # Package to numerically differentiate
@everywhere using Dierckx # Package for interpolation
# @everywhere using SharedArrays
@everywhere include("funcs.jl")

using CSV
using Dates

index_to_append = ARGS[1]

print("\n--- Loading Data ----\n")

svi_filepath = string("data/raw_data/svi_params_", index_to_append, ".csv")
svi_data = DataFrame(CSV.File(svi_filepath))


svi_data = svi_data[svi_data.opt_out .== "FTOL_REACHED", :]

####################################################################################################
# There is one observation that raises an error, remove it for now manually. Deal with it later
# secid: 110472, obs_date = 2021-05-27, exp_date = 2021-06-04
function filter_func(secid, obs_date, exp_date)::Bool
	!((secid == 110472) && (obs_date == Date("2021-05-27")) && (exp_date == Date("2021-06-04")))
end

svi_data = filter([:secid, :obs_date, :exp_date] => filter_func, svi_data)
####################################################################################################

num_options = size(svi_data)[1]

print(string("\n--- Number of options to estimate parameters for ", num_options, " ---\n"))

# Generating SVIParams structs from inputs:
svi_arr = Array{OptionData, 1}(undef, num_options)
m_arr = svi_data.m
sigma_arr = svi_data.sigma
a_arr = svi_data.a
b_arr = svi_data.b
obj_arr = svi_data.obj
opt_out_arr = svi_data.opt_out
spot_arr = svi_data.spot
r_arr = svi_data.r
F_arr = svi_data.F
T_arr = svi_data.T
sigma_NTM_arr = svi_data.sigma_NTM
max_K_arr = svi_data.max_K
min_K_arr = svi_data.min_K
svi_arr = map(i -> SVIParams(m_arr[i], sigma_arr[i], 0.0, a_arr[i], b_arr[i],
                             obj_arr[i], opt_out_arr[i]), 1:num_options)

print("\n--- Estimating parameters ---\n")
print("\n--- First Pass ---\n")
@time tmp = pmap(estimate_parameters, spot_arr[1:2], r_arr[1:2], F_arr[1:2], T_arr[1:2],
                sigma_NTM_arr[1:2], min_K_arr[1:2], max_K_arr[1:2],  svi_arr[1:2])

print("\n--- Second Pass ---\n")
@time ests = pmap(estimate_parameters, spot_arr, r_arr,
                  F_arr, T_arr, sigma_NTM_arr,
                  min_K_arr, max_K_arr, svi_arr)

print("\n--- Outputting Data ---\n")
df_out = DataFrame(
              secid = svi_data.secid,
              date = svi_data.obs_date,
              T = T_arr,
              V_in_sample = map(x -> x[1], ests),
              IV_in_sample = map(x -> x[2], ests),
              V_clamp = map(x -> x[3], ests),
              IV_clamp = map(x -> x[4], ests),
              rn_prob_sigma = map(x -> x[5], ests),
              rn_prob_2sigma = map(x -> x[6], ests),
              rn_prob_20 = map(x -> x[7], ests),
              rn_prob_40 = map(x -> x[8], ests),
              rn_prob_60 = map(x -> x[9], ests),
              rn_prob_80 = map(x -> x[10], ests)
            )

CSV.write(string("data/output/var_ests_", index_to_append, ".csv"), df_out)


# for irow in 1:size(svi_data)[1]
# 	@show irow
# 	row = svi_data[irow,:];
# 	estimate_parameters(
# 		row[:spot],
# 		row[:r],
# 		row[:F],
# 		row[:T],
# 		row[:sigma_NTM],
# 		row[:min_K],
# 		row[:max_K],
# 		svi_arr[irow]);
# end


# # row 23467. secid: 110472, obs_date = 2021-05-27, exp_date = 2021-06-04
# irow = 23467
# row = svi_data[irow,:];
# params = estimate_parameters(
# 	row[:spot],
# 	row[:r],
# 	row[:F],
# 	row[:T],
# 	row[:sigma_NTM],
# 	row[:min_K],
# 	row[:max_K],
# 	svi_arr[irow]);


# spot = row[:spot]
# r = row[:r]
# F = row[:F]
# T = row[:T]
# interp_params = svi_arr[irow]
# min_K = row[:min_K]
# max_K = row[:max_K]

# calc_option_value_put = K -> calc_option_value(spot, r, F, T, interp_params, K, min_K, max_K, "Put")
# calc_option_value_call = K -> calc_option_value(spot, r, F, T, interp_params, K, min_K, max_K, "Call")

# IV1_raw = K -> calc_option_value_call(K)/K^2
# IV1 = t -> IV1_raw(spot + t/(1-t))/(1-t)^2

# IV2_raw = K -> calc_option_value_put(K)/K^2
# IV2 = t -> IV2_raw(spot * t) * spot

# # integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1, maxevals = 100000)[1] + hquadrature(IV2, 0, 1, maxevals = 100000)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
# hquadrature(IV2, 0.1, 1, maxevals = 100000)[1]

# trange = 0.01:0.001:1.0;
# IV2_arr = zeros(length(trange));
# for i in 1:length(trange)
# 	IV2_arr[i] = IV2(trange[i])
# end

# Plots.plot(IV2_arr)
