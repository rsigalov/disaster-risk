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

svi_data = CSV.read(string("data/raw_data/svi_params_", index_to_append, ".csv"); datarow = 2, delim = ",")
svi_data = svi_data[svi_data.opt_out .== "FTOL_REACHED", :]

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
sigma_NTM_arr = svi_data.sigma_NTM_new # Using new Near-The-Money sigma for estimation
max_K_arr = svi_data.max_K
min_K_arr = svi_data.min_K
svi_arr = map(i -> SVIParams(m_arr[i], sigma_arr[i], 0.0, a_arr[i], b_arr[i],
                             obj_arr[i], opt_out_arr[i]), 1:num_options)

print("\n--- Estimating parameters ---\n")
print("\n--- First Pass ---\n")
@time tmp = pmap(estimate_parameters_rn_prob, spot_arr[1:2], r_arr[1:2], F_arr[1:2], T_arr[1:2],
    sigma_NTM_arr[1:2],min_K_arr[1:2], max_K_arr[1:2], svi_arr[1:2])

print("\n--- Second Pass ---\n")
@time ests = pmap(estimate_parameters_rn_prob, spot_arr, r_arr,
                  F_arr, T_arr, sigma_NTM_arr,
                  min_K_arr, max_K_arr, svi_arr)

print("\n--- Outputting Data ---\n")
df_out = DataFrame(secid = svi_data.secid,
              date = svi_data.obs_date,
              T = T_arr,
              rn_prob_sigma = map(x -> x[1], ests),
              rn_prob_2sigma = map(x -> x[2], ests),
              rn_prob_20 = map(x -> x[3], ests),
              rn_prob_40 = map(x -> x[4], ests),
              rn_prob_60 = map(x -> x[5], ests),
              rn_prob_80 = map(x -> x[6], ests))

CSV.write(string("data/output_rn_prob/var_ests_", index_to_append, ".csv"), df_out)

print("\n--- Done ---\n")
