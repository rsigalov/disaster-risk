using DataFrames
using NLopt # Package to perform numerical optiimization
using NLsolve
using LinearAlgebra # Package with some useful functions
using Distributions # Package for normal CDF
using HCubature # Package to numerically integrate
using ForwardDiff # Package to numerically differentiate
using Dierckx # Package for interpolation
include("funcs.jl")
include("simulation_funcs.jl")
cd("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")
using CSV
using Dates

using Plots
using PyPlot

# Fixing parameters for simulation:
lambda_disaster = 0.05
mean_disaster = -0.8
sigma_disaster = 0.1
lambda_small = 20
mean_small = -0.005
sigma_small = 0.01

low_strike = 0.8
high_strike = 1.6

# Additional parameters:
theta = 0.1^2 # unconditional mean of Variance
k = 5.33 # mean reversion speed
sigma = 0.14 # volatility of volatility
rho = -0.05 # correlation of volatility draws and price innovation draws
T = 1 # maturity
spot = 100 # spot price
r = 0.025
min_K = low_strike * 100
max_K = high_strike * 100
F = exp(r * T) * spot


################################################
# Simulation part
################################################
# 1. Simulate paths
process = process_parameters(
    theta, # theta
    k, # k
    sigma, # sigma
    rho, # rho
    r,  # r
    T, # T
    spot, # spot
    lambda_small, # lambda_small
    mean_small, # mean_jump_small
    sigma_small, # sigma_jump_small
    lambda_disaster, # lambda_disaster
    mean_disaster, # mean_jump_disaster
    sigma_disaster, # sigma_jump_disaster
)

S1 = simulate_paths(process, 10, 10000, 1000)

# 2. Simulating option prices
strike_list, put_values, call_values = calculate_option_values(S1, process, spot * low_strike, spot * high_strike, 20)
put_impl_vol, call_impl_vol, impl_vol = calculate_implied_vol(process, strike_list, put_values, call_values)

# 3. Fitting SVI into simulated option prices:
option, svi = create_option_and_fit_svi(process, strike_list, impl_vol)

function V_IV_limits_svi(low_limit, high_limit)
    calc_V_IV_D(spot, r, F, T, svi, min_K, max_K, low_limit, high_limit, false)
end

function V_IV_limits_clamped(low_limit, high_limit)
    calc_V_IV_D(spot, r, F, T, svi, min_K, max_K, low_limit, high_limit, true)
end

# Calculating V and IV for a set of low and high limits:
high_limit_list = LinRange(300, 30, 51)

ests_svi = map(x -> V_IV_limits_svi(0, x), high_limit_list)
ests_clamped = map(x -> V_IV_limits_clamped(0, x), high_limit_list)

# Calculating D for each extrapolation:
D_svi = map(x -> x[1] - x[2], ests_svi)
D_clamp = map(x -> x[1] - x[2], ests_clamped)

########################################################################
# Calculating actual D coming from each jump process
########################################################################
function slope(mu, sigma)
    return 2 * (1 + mu + 0.5*(mu^2 + sigma^2) - exp(mu + 0.5sigma^2))
end

# D from disaster
D_disaster = slope(mean_disaster, sigma_disaster) * lambda_disaster
# D from small jumps
D_small = slope(mean_small, sigma_small) * lambda_small
# Overall D
D_full = D_disaster + D_small

########################################################################
# Plotting
########################################################################

Plots.plot(high_limit_list, D_svi, label = "SVI",
    xlabel = "Higher Limit of Integration", legend = :bottom)
Plots.plot!(high_limit_list, D_clamp, label = "Clamp")
Plots.hline!([D_full], label = "Full D")
Plots.hline!([D_disaster], label = "Disaster D")
Plots.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/D_high_limit_sens_case_3_high_strike.pdf")
