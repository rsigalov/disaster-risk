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
# cd("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")
using CSV
using Dates

using Plots
using PyPlot

mean_small = -0.005
sigma_small = 0.01
case = "3"


################################################################
# Main simulations code:
################################################################

function do_simulation(lambda_small, lambda_disaster)
    process = process_parameters(
        0.10^2, # theta
        5.33, # k
        0.14, # sigma
        -0.05, # rho
        0.025,  # r
        1, # T
        100, # spot
        lambda_small, # lambda_small
        mean_small, # mean_jump_small
        sigma_small, # sigma_jump_small
        lambda_disaster, # lambda_disaster
        -0.8, # mean_jump_disaster
        0.1, # sigma_jump_disaster
    )

    S1 = simulate_paths(process, 10, 10000, 1000)

    return S1, process
end

function do_all_estimations(S1, process, low_strike)
    spot = 100

    strike_list, put_values, call_values = calculate_option_values(S1, process, spot * low_strike, spot * 1.6, 20)
    put_impl_vol, call_impl_vol, impl_vol = calculate_implied_vol(process, strike_list, put_values, call_values)
    option, svi = create_option_and_fit_svi(process, strike_list, impl_vol)
    D_ests = estimate_parameters_return_Ds(option, svi)
    prob_drop_full, prob_drop_clamp = calculate_rn_prb(option, svi)
    act_prob_drop = sum(S1 .<= 0.5 * spot)/length(S1)

    return strike_list, put_values, call_values, put_impl_vol, call_impl_vol, impl_vol, option, svi, D_ests,
        act_prob_drop, prob_drop_full, prob_drop_clamp
end

# List of parameters:
lambda_small_base = 20
lambda_disaster_base = 0.05
lambda_small_list = LinRange(5, 40, 10)
lambda_disaster_list = LinRange(0.01, 0.2, 10)
low_strike_list = [0.25, 0.4, 0.6, 0.8]

sims_vary_small = map(x -> do_simulation(x, lambda_disaster_base), lambda_small_list)
sims_vary_disaster = map(x -> do_simulation(lambda_small_base, x), lambda_disaster_list)

# Plotting histograms:
# Plots.histogram(sims_vary_small[1][1], alpha = 0.5, label = string(lambda_small_list[1]), size = (600, 400))
# Plots.histogram!(sims_vary_small[5][1], alpha = 0.5, label = string(lambda_small_list[5]))
# Plots.histogram!(sims_vary_small[10][1], alpha = 0.5, label = string(lambda_small_list[10]))
# Plots.savefig("write-up-files/Simulations/images/density_fix_disaster_intensity_alt.pdf")
#
# Plots.histogram(sims_vary_disaster[1][1], alpha = 0.5, label = string(lambda_disaster_list[1]), size = (600, 400))
# Plots.histogram!(sims_vary_disaster[5][1], alpha = 0.5, label = string(lambda_disaster_list[2]))
# Plots.histogram!(sims_vary_disaster[10][1], alpha = 0.5, label = string(lambda_disaster_list[3]))
# Plots.savefig("write-up-files/Simulations/images/density_fix_small_intensity_alt.pdf")

# Estimating stuff:
res_vary_small_1 = map(x -> do_all_estimations(x[1], x[2], low_strike_list[1]), sims_vary_small)
res_vary_small_2 = map(x -> do_all_estimations(x[1], x[2], low_strike_list[2]), sims_vary_small)
res_vary_small_3 = map(x -> do_all_estimations(x[1], x[2], low_strike_list[3]), sims_vary_small)
res_vary_small_4 = map(x -> do_all_estimations(x[1], x[2], low_strike_list[4]), sims_vary_small)

res_vary_disaster_1 = map(x -> do_all_estimations(x[1], x[2], low_strike_list[1]), sims_vary_disaster)
res_vary_disaster_2 = map(x -> do_all_estimations(x[1], x[2], low_strike_list[2]), sims_vary_disaster)
res_vary_disaster_3 = map(x -> do_all_estimations(x[1], x[2], low_strike_list[3]), sims_vary_disaster)
res_vary_disaster_4 = map(x -> do_all_estimations(x[1], x[2], low_strike_list[4]), sims_vary_disaster)

# Plotting how different variables compare:
# Plots.scatter(lambda_small_list./lambda_small_base, map(x -> res_vary_small_1[x][10][2], 1:10))
# Plots.scatter!(lambda_small_list./lambda_small_base, map(x -> res_vary_disaster_1[x][10][2], 1:10))

# Doing a dataset with all results:
df_out = DataFrame(measure = ["M", "F"], min_strike = [1.0, 2.0],
                   var_to_vary = ["M", "F"], lambda_level = [1.0, 2.0], D = [1.0, 2.0])

lambda_level_small = lambda_small_list./lambda_small_base
lambda_level_disaster = lambda_disaster_list./lambda_disaster_base
measure_name_list = ["D", "D in sample", "D bound", "D puts", "D bounds puts",
                     "D deep puts", "D bound deep puts", "D clamp", "D clamp puts",
                     "D clamp deep puts"]

for i_measure = 1:10
    @show i_measure
    df_to_append = DataFrame(measure = measure_name_list[i_measure], min_strike = low_strike_list[1],
        var_to_vary = "Small Intensity", lambda_level = lambda_level_small, D = map(x -> res_vary_small_1[x][9][i_measure], 1:10))
    append!(df_out, df_to_append)

    df_to_append = DataFrame(measure = measure_name_list[i_measure], min_strike = low_strike_list[2],
        var_to_vary = "Small Intensity", lambda_level = lambda_level_small, D = map(x -> res_vary_small_2[x][9][i_measure], 1:10))
    append!(df_out, df_to_append)

    df_to_append = DataFrame(measure = measure_name_list[i_measure], min_strike = low_strike_list[3],
        var_to_vary = "Small Intensity", lambda_level = lambda_level_small, D = map(x -> res_vary_small_3[x][9][i_measure], 1:10))
    append!(df_out, df_to_append)

    df_to_append = DataFrame(measure = measure_name_list[i_measure], min_strike = low_strike_list[4],
        var_to_vary = "Small Intensity", lambda_level = lambda_level_small, D = map(x -> res_vary_small_4[x][9][i_measure], 1:10))
    append!(df_out, df_to_append)

    df_to_append = DataFrame(measure = measure_name_list[i_measure], min_strike = low_strike_list[1],
        var_to_vary = "Disaster Intensity", lambda_level = lambda_level_disaster, D = map(x -> res_vary_disaster_1[x][9][i_measure], 1:10))
    append!(df_out, df_to_append)

    df_to_append = DataFrame(measure = measure_name_list[i_measure], min_strike = low_strike_list[2],
        var_to_vary = "Disaster Intensity", lambda_level = lambda_level_disaster, D = map(x -> res_vary_disaster_2[x][9][i_measure], 1:10))
    append!(df_out, df_to_append)

    df_to_append = DataFrame(measure = measure_name_list[i_measure], min_strike = low_strike_list[3],
        var_to_vary = "Disaster Intensity", lambda_level = lambda_level_disaster, D = map(x -> res_vary_disaster_3[x][9][i_measure], 1:10))
    append!(df_out, df_to_append)

    df_to_append = DataFrame(measure = measure_name_list[i_measure], min_strike = low_strike_list[4],
        var_to_vary = "Disaster Intensity", lambda_level = lambda_level_disaster, D = map(x -> res_vary_disaster_4[x][9][i_measure], 1:10))
    append!(df_out, df_to_append)
end

df_out = df_out[3:end, :]

CSV.write(string("simulations_results_case_", case, ".csv"), df_out)


####################################################################
# Calculating probability of a large decline for each SVI and
# comparing it with the actual:
prob_actual = map(x -> x[10], res_vary_disaster_1)
prob_full = map(x -> x[11], res_vary_disaster_1)
prob_clamp = map(x -> x[12], res_vary_disaster_1)

df_out_prob = DataFrame(measure = ["M", "F"], min_strike = [1.0, 2.0],
                var_to_vary = ["M", "F"], lambda_level = [1.0, 2.0], prob = [1.0, 2.0])

res_vary_small = (res_vary_small_1, res_vary_small_2, res_vary_small_3, res_vary_small_4)
res_vary_disaster = (res_vary_disaster_1, res_vary_disaster_2, res_vary_disaster_3, res_vary_disaster_4)

for i_min_strike = 1:4

    df_to_append = DataFrame(measure = "Full", min_strike = low_strike_list[i_min_strike],
        var_to_vary = "Small Intensity Actual", lambda_level = lambda_level_small, prob = map(x -> res_vary_small[i_min_strike][x][10], 1:10))
    append!(df_out_prob, df_to_append)

    df_to_append = DataFrame(measure = "Full", min_strike = low_strike_list[i_min_strike],
        var_to_vary = "Disaster Intensity Actual", lambda_level = lambda_level_small, prob = map(x -> res_vary_disaster[i_min_strike][x][10], 1:10))
    append!(df_out_prob, df_to_append)

    df_to_append = DataFrame(measure = "Full", min_strike = low_strike_list[i_min_strike],
        var_to_vary = "Small Intensity", lambda_level = lambda_level_small, prob = map(x -> res_vary_small[i_min_strike][x][11], 1:10))
    append!(df_out_prob, df_to_append)

    df_to_append = DataFrame(measure = "Full", min_strike = low_strike_list[i_min_strike],
        var_to_vary = "Disaster Intensity", lambda_level = lambda_level_small, prob = map(x -> res_vary_disaster[i_min_strike][x][11], 1:10))
    append!(df_out_prob, df_to_append)

    df_to_append = DataFrame(measure = "Clamp", min_strike = low_strike_list[i_min_strike],
        var_to_vary = "Small Intensity Actual", lambda_level = lambda_level_small, prob = map(x -> res_vary_small[i_min_strike][x][10], 1:10))
    append!(df_out_prob, df_to_append)

    df_to_append = DataFrame(measure = "Clamp", min_strike = low_strike_list[i_min_strike],
        var_to_vary = "Disaster Intensity Actual", lambda_level = lambda_level_small, prob = map(x -> res_vary_disaster[i_min_strike][x][10], 1:10))
    append!(df_out_prob, df_to_append)

    df_to_append = DataFrame(measure = "Clamp", min_strike = low_strike_list[i_min_strike],
        var_to_vary = "Small Intensity", lambda_level = lambda_level_small, prob = map(x -> res_vary_small[i_min_strike][x][12], 1:10))
    append!(df_out_prob, df_to_append)

    df_to_append = DataFrame(measure = "Clamp", min_strike = low_strike_list[i_min_strike],
        var_to_vary = "Disaster Intensity", lambda_level = lambda_level_small, prob = map(x -> res_vary_disaster[i_min_strike][x][12], 1:10))
    append!(df_out_prob, df_to_append)

end

df_out_prob = df_out_prob[3:end, :]

CSV.write(string("simulation_results_prob_case_", case, ".csv"), df_out_prob)


####################################################
# Plotting distribution of price at time T = 1 for
# different sizes of small jumps
####################################################

function do_simulation(lambda_small, lambda_disaster, mean_small, std_small)
    process = process_parameters(
        0.10^2, # theta
        5.33, # k
        0.14, # sigma
        -0.05, # rho
        0.025,  # r
        1, # T
        100, # spot
        lambda_small, # lambda_small
        mean_small, # mean_jump_small
        std_small, # sigma_jump_small
        lambda_disaster, # lambda_disaster
        -0.8, # mean_jump_disaster
        0.1, # sigma_jump_disaster
    )

    S1 = simulate_paths(process, 10, 10000, 1000)

    return S1, process
end

### Case I, small jump ~ N(-0.05, 0.05)
comp_sims = map(x -> do_simulation(x, 0, -0.05, 0.05)[1], [0, 5, 20, 40])

# Plotting histograms for each
Plots.histogram(comp_sims[1], alpha = 0.5, label = string(0),
    size = (700, 500), title = "Small Jump log(r) ~ N(-0.05, 0.05), no disaster")
Plots.histogram!(comp_sims[2], alpha = 0.5, label = string(5))
Plots.histogram!(comp_sims[3], alpha = 0.5, label = string(20))
Plots.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/s1_distribution_case_1_no_disaster.pdf")

comp_sims_2 = map(x -> do_simulation(x, 0.05, -0.05, 0.05)[1], [0, 5, 20])

Plots.histogram(comp_sims_2[1], alpha = 0.5, label = string(0),
    size = (700, 500), title = "Small Jump log(r) ~ N(-0.05, 0.05), disaster log(r) ~ N(-0.8, 0.1)")
Plots.histogram!(comp_sims_2[2], alpha = 0.5, label = string(5))
Plots.histogram!(comp_sims_2[3], alpha = 0.5, label = string(20))
Plots.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/s1_distribution_case_1_disaster.pdf")

### Case II, small jump ~ N(-0.005, 0.05)
comp_sims_3 = map(x -> do_simulation(x, 0, -0.005, 0.05)[1], [0, 5, 20])

Plots.histogram(comp_sims_3[1], alpha = 0.5, label = string(0),
    size = (700, 500), title = "Small Jump log(r) ~ N(-0.005, 0.05), no disaster")
Plots.histogram!(comp_sims_3[2], alpha = 0.5, label = string(5))
Plots.histogram!(comp_sims_3[3], alpha = 0.5, label = string(20))
Plots.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/s1_distribution_case_2_no_disaster.pdf")

comp_sims_4 = map(x -> do_simulation(x, 0.05, -0.005, 0.05)[1], [0, 5, 20])

Plots.histogram(comp_sims_4[1], alpha = 0.5, label = string(0),
    size = (700, 500), title = "Small Jump log(r) ~ N(-0.005, 0.05), disaster log(r) ~ N(-0.8, 0.1)")
Plots.histogram!(comp_sims_4[2], alpha = 0.5, label = string(5))
Plots.histogram!(comp_sims_4[3], alpha = 0.5, label = string(20))
Plots.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/s1_distribution_case_2_disaster.pdf")

### Case I, small jump ~ N(-0.005, 0.01)
comp_sims_5 = map(x -> do_simulation(x, 0, -0.005, 0.01)[1], [0, 5, 20])

Plots.histogram(comp_sims_5[1], alpha = 0.5, label = string(0),
    size = (700, 500), title = "Small Jump log(r) ~ N(-0.005, 0.01), no disaster")
Plots.histogram!(comp_sims_5[2], alpha = 0.5, label = string(5))
Plots.histogram!(comp_sims_5[3], alpha = 0.5, label = string(20))
Plots.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/s1_distribution_case_3_no_disaster.pdf")

comp_sims_6 = map(x -> do_simulation(x, 0.05, -0.005, 0.01)[1], [0, 5, 20])

Plots.histogram(comp_sims_6[1], alpha = 0.5, label = string(0),
    size = (700, 500), title = "Small Jump log(r) ~ N(-0.005, 0.01), disaster log(r) ~ N(-0.8, 0.1)")
Plots.histogram!(comp_sims_6[2], alpha = 0.5, label = string(5))
Plots.histogram!(comp_sims_6[3], alpha = 0.5, label = string(20))
Plots.savefig("/Users/rsigalov/Dropbox/2019_Revision/Writing/Simulations/images/s1_distribution_case_3_disaster.pdf")



# Comparing D calculated from a discrete set of simulated option prices to
#   1) D calculated from a continuum of option prices
#   2) Theoretical do
#   3) D calculated as

function different_Ds(lambda_small, mu_small, sigma_small,
                      lambda_disaster, mu_disaster, sigma_disaster)

    process = process_parameters(
        0.10^2, # theta
        5.33, # k
        0.14, # sigma
        -0.05, # rho
        0.025,  # r
        1, # T
        100, # spot
        lambda_small, # lambda_small
        mu_small, # mean_jump_small
        sigma_small, # sigma_jump_small
        lambda_disaster, # lambda_disaster
        mu_disaster, # mean_jump_disaster
        sigma_disaster, # sigma_jump_disaster
    )

    S1, int_elem = simulate_paths(process, 10, 10000, 1000)
    Plots.histogram(S1)

    # Calculating D using a continous set of strikes:
    range_S1 = maximum(S1) - minimum(S1)
    min_K = minimum(S1) + 0.025 * range_S1
    max_K = maximum(S1) - 0.025 * range_S1
    low_limit = 0
    high_limit = 1000

    strike_list, put_values, call_values = calculate_option_values(S1, process, min_K, max_K, 1000)
    put_impl_vol, call_impl_vol, impl_vol = calculate_implied_vol(process, strike_list, put_values, call_values)
    option, svi = create_option_and_fit_svi(process, strike_list, impl_vol)
    D_ests = estimate_parameters_return_Ds(option, svi)

    # Calculating Variation and Integrated Variation using explicit formulas:
    V_explicit_simulation = var(log.(S1))
    IV_explicit_simulation = 2*(mean(int_elem) - mean(log.(S1./process.spot)))
    D_explicit_simulation = V_explicit_simulation - IV_explicit_simulation

    # Calculating V and IV using theoreticla results:
    function DD(mu, sigma, lambda)
        return 2 * (1 + mu + 0.5 * (mu^2 + sigma^2) - exp(mu + 0.5*sigma^2)) * lambda
    end

    D_theory = DD(process.mu_small, process.sigma_small, process.lambda_small) +
        DD(process.mu_disaster, process.sigma_disaster, process.lambda_disaster)

    function put_value_dist(K, S1, process)
        return exp(-process.r * process.T) .* mean(max.(0, K .- S1))
    end

    function call_value_dist(K, S1, process)
        return  exp(-process.r * process.T) .* mean(max.(0, S1 .- K))
    end

    spot = process.spot
    r = process.r
    T = process.T

    # 1. First define call and put option prices as functions of the strike:
    calc_option_value_put = K -> put_value_dist(K, S1, process)
    calc_option_value_call = K -> call_value_dist(K, S1, process)
    Plots.plot(strike_list, calc_option_value_call.(strike_list))
    Plots.plot!(strike_list, calc_option_value_put.(strike_list))

    # calc_option_value_put = K -> calc_option_value(option, svi, K, "Put")
    # calc_option_value_call = K -> calc_option_value(option, svi, K, "Call")
    # Plots.plot!(strike_list, calc_option_value_call.(strike_list))
    # Plots.plot!(strike_list, calc_option_value_put.(strike_list))

    # These integrals contain call(strike) function
    V1_raw = K -> 2 * (1 - log(K/spot)) * calc_option_value_call(K)/K^2
    W1_raw = K -> (6 * log(K/spot) - 3 * (log(K/spot))^2) * calc_option_value_call(K)/K^2
    X1_raw = K -> (12 * log(K/spot)^2 - 4 * log(K/spot)^3) * calc_option_value_call(K)/K^2

    # These integrals contain put(strike) function
    V2_raw = K -> 2 * (1 + log(spot/K)) * calc_option_value_put(K)/K^2
    W2_raw = K -> (6 * log(spot/K) + 3 * log(spot/K)^2) * calc_option_value_put(K)/K^2
    X2_raw = K -> (12 * log(spot/K)^2 + 4 * log(spot/K)^3) * calc_option_value_put(K)/K^2

    IV1_raw = K -> calc_option_value_call(K)/K^2
    IV1 = t -> IV1_raw(spot + t/(1-t))/(1-t)^2

    IV2 = K -> calc_option_value_put(K)/K^2

    integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1)[1] +
                                             hquadrature(IV2, 0, spot)[1] -
                                             exp(-r*T)*(exp(r*T)-1-r*T))

    # Modifying integrands to account for infinite upper integration limit
    V1 = t -> V1_raw(spot + t/(1-t))/(1-t)^2
    W1 = t -> W1_raw(spot + t/(1-t))/(1-t)^2
    X1 = t -> X1_raw(spot + t/(1-t))/(1-t)^2

    V = hquadrature(V1, 0, 1)[1] + hquadrature(V2_raw, low_limit, spot)[1]
    W = hquadrature(W1, 0, 1)[1] - hquadrature(W2_raw, low_limit, spot)[1]
    X = hquadrature(X1, 0, 1)[1] + hquadrature(X2_raw, low_limit, spot)[1]

    mu = exp(r*T) - 1 - exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

    variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)

    D_cont_strikes  = variation - integrated_variation

    return D_theory, D_explicit_simulation, D_cont_strikes, D_ests[1]

end

lambda_disaster_list = [0.05, 0.01, 0.2]

case_1_ests = map(x -> different_Ds(20, -0.05, 0.05, x, -0.8, 0.1), lambda_disaster_list)
case_2_ests = map(x -> different_Ds(20, -0.005, 0.05, x, -0.8, 0.1), lambda_disaster_list)
case_3_ests = map(x -> different_Ds(20, -0.005, 0.01, x, -0.8, 0.1), lambda_disaster_list)

# Ploting stuff
Plots.scatter(lambda_disaster_list, map(x -> x[1], case_1_ests), label = "Theoretical", ylims = (0,0.04),
    legend = :bottomright, alpha = 0.5, xlabel = "Disaster Intensity",
    title = "Case I, small jumps log(r) ~ N(-0.05, 0.05)")
Plots.scatter!(lambda_disaster_list, map(x -> x[2], case_1_ests), label = "Explicit Simulation", alpha = 0.5)
Plots.scatter!(lambda_disaster_list, map(x -> x[3], case_1_ests), label = "Continous Strikes", alpha = 0.5)
Plots.scatter!(lambda_disaster_list, map(x -> x[4], case_1_ests), label = "1000 options interpolated", alpha = 0.5)

Plots.scatter(lambda_disaster_list, map(x -> x[1], case_2_ests), label = "Theoretical", ylims = (0,0.04),
    legend = :bottomright, alpha = 0.5, xlabel = "Disaster Intensity",
    title = "Case II, small jumps log(r) ~ N(-0.005, 0.05)")
Plots.scatter!(lambda_disaster_list, map(x -> x[2], case_2_ests), label = "Explicit Simulation", alpha = 0.5)
Plots.scatter!(lambda_disaster_list, map(x -> x[3], case_2_ests), label = "Continous Strikes", alpha = 0.5)
Plots.scatter!(lambda_disaster_list, map(x -> x[4], case_2_ests), label = "1000 options interpolated", alpha = 0.5)

Plots.scatter(lambda_disaster_list, map(x -> x[1], case_3_ests), label = "Theoretical", ylims = (0,0.04),
    legend = :bottomright, alpha = 0.5, xlabel = "Disaster Intensity",
    title = "Case III, small jumps log(r) ~ N(-0.005, 0.01)")
Plots.scatter!(lambda_disaster_list, map(x -> x[2], case_3_ests), label = "Explicit Simulation", alpha = 0.5)
Plots.scatter!(lambda_disaster_list, map(x -> x[3], case_3_ests), label = "Continous Strikes", alpha = 0.5)
Plots.scatter!(lambda_disaster_list, map(x -> x[4], case_3_ests), label = "1000 options interpolated", alpha = 0.5)




####################################################
# Old stuff

#
# ####################################################
# # Plotting some implied vols and SVI curves:
# option, svi = res_vary_small_1[1][8:9]
#
# log_moneyness = log.(option.strikes/option.spot)
# impl_vol = option.impl_vol
#
# range_log_moneyness = log_moneyness[end] - log_moneyness[1]
# plot_range = LinRange(log_moneyness[1] - range_log_moneyness*0.10,
#                       log_moneyness[end] + range_log_moneyness*0.10, 1000);
#
# Plots.scatter(log_moneyness, impl_vol.^2, label = "5.0", size = (600,400))
# Plots.plot!(plot_range, svi_smile(plot_range, svi.m,
#                                   svi.sigma, svi.rho,
#                                   svi.a, svi.b), label = "5.0")
#
# option, svi = res_vary_small_1[5][8:9]
#
# log_moneyness = log.(option.strikes/option.spot)
# impl_vol = option.impl_vol
#
# range_log_moneyness = log_moneyness[end] - log_moneyness[1]
# plot_range = LinRange(log_moneyness[1] - range_log_moneyness*0.10,
#                     log_moneyness[end] + range_log_moneyness*0.10, 1000);
#
# Plots.scatter!(log_moneyness, impl_vol.^2, label = "20.56")
# Plots.plot!(plot_range, svi_smile(plot_range, svi.m,
#                                 svi.sigma, svi.rho,
#                                 svi.a, svi.b), label = "20.56")
#
# option, svi = res_vary_small_1[10][8:9]
#
# log_moneyness = log.(option.strikes/option.spot)
# impl_vol = option.impl_vol
#
# range_log_moneyness = log_moneyness[end] - log_moneyness[1]
# plot_range = LinRange(log_moneyness[1] - range_log_moneyness*0.10,
#                     log_moneyness[end] + range_log_moneyness*0.10, 1000);
#
# Plots.scatter!(log_moneyness, impl_vol.^2, label = "40")
# Plots.plot!(plot_range, svi_smile(plot_range, svi.m,
#                                 svi.sigma, svi.rho,
#                                 svi.a, svi.b), label = "40")
# Plots.xaxis!("log-moneyness")
# Plots.yaxis!("BS implied variance")
# Plots.savefig("write-up-files/Simulations/images/svi_fit_fix_disaster_intensity_alt.pdf")
#
# ####################################################
# # Plotting some implied vols and SVI curves:
# option, svi = res_vary_disaster_1[1][8:9]
#
# log_moneyness = log.(option.strikes/option.spot)
# impl_vol = option.impl_vol
#
# range_log_moneyness = log_moneyness[end] - log_moneyness[1]
# plot_range = LinRange(log_moneyness[1] - range_log_moneyness*0.10,
#                       log_moneyness[end] + range_log_moneyness*0.10, 1000);
#
# Plots.scatter(log_moneyness, impl_vol.^2, label = "0.01", size = (600,400))
# Plots.plot!(plot_range, svi_smile(plot_range, svi.m,
#                                   svi.sigma, svi.rho,
#                                   svi.a, svi.b), label = "0.01")
#
# option, svi = res_vary_disaster_1[5][8:9]
#
# log_moneyness = log.(option.strikes/option.spot)
# impl_vol = option.impl_vol
#
# range_log_moneyness = log_moneyness[end] - log_moneyness[1]
# plot_range = LinRange(log_moneyness[1] - range_log_moneyness*0.10,
#                     log_moneyness[end] + range_log_moneyness*0.10, 1000);
#
# Plots.scatter!(log_moneyness, impl_vol.^2, label = "0.094")
# Plots.plot!(plot_range, svi_smile(plot_range, svi.m,
#                                 svi.sigma, svi.rho,
#                                 svi.a, svi.b), label = "0.094")
#
# option, svi = res_vary_disaster_1[10][8:9]
#
# log_moneyness = log.(option.strikes/option.spot)
# impl_vol = option.impl_vol
#
# range_log_moneyness = log_moneyness[end] - log_moneyness[1]
# plot_range = LinRange(log_moneyness[1] - range_log_moneyness*0.10,
#                     log_moneyness[end] + range_log_moneyness*0.10, 1000);
#
# Plots.scatter!(log_moneyness, impl_vol.^2, label = "0.2")
# Plots.plot!(plot_range, svi_smile(plot_range, svi.m,
#                                 svi.sigma, svi.rho,
#                                 svi.a, svi.b), label = "0.2")
# Plots.xaxis!("log-moneyness")
# Plots.yaxis!("BS implied variance")
# Plots.savefig("write-up-files/Simulations/images/svi_fit_fix_small_intensity_alt.pdf")
#
# #########################################################
# # Testing stuff.





# # General parameters:
# num_chunks = 10
# size_chunk = 10000
# num_paths = num_chunks * size_chunk # 100 chunks of 10k paths:
# len_path = 1000
# T = 1
# dt = T/len_path
#
# # Parameters for simulations:
# # lambda_small = 20 # intensity for small jumps
# lambda_small = 20
# # lambda_disaster = 0.01 # intensity for the disaster jumps
# lambda_disaster = 0.05
#
# # lambda_small = 1.65
# # lambda_disaster = 0
#
# # mu_small = -0.0385
# # sigma_small = 0.02
#
# mu_small = -0.005
# sigma_small = 0.05
#
# mu_disaster = -0.6
# sigma_disaster = 0.1
#
# # Parameters for stochastic volatility part:
# theta = 0.10^2 # unconditional mean of volatility process
# k = 5.33 # mean-reversion parameter for volatility
# sigma = 0.14 # volatility of process for volatility
# rho = -0.05 # correlation between dW1 and dW2
# Sigma = [1 rho;
#          rho 1]
#
# # Stock parameters:
# r = 0.025
# spot = 100
# forward = exp(r*T) * spot
#
# # Don't need to save the path, hence, will work in chunks and
# # not keep in memory all the paths:
# S1 = zeros(num_paths) # array to store all final values of the price
#                       # on which option pricing will be based
#
# dW_normal = Distributions.MvNormal(Sigma)
# # Simulating Poissong jumps:
# poisson_small = Poisson(lambda_small * dt)
# # jump_small = TruncatedNormal(mu_small, sigma_small, -0.1, 0.1)
# jump_small = Normal(mu_small, sigma_small)
# # jump_small = TruncatedNormal(0, 0.05, -0.9, 0.9)
# poisson_disaster = Poisson(lambda_disaster * dt)
# # jump_disaster = TruncatedNormal(-0.4, 1, -0.75, -0.25)
# jump_disaster = Normal(mu_disaster, sigma_disaster)
#
# # comp_small = lambda_small * mean(jump_small)
# comp_small = lambda_small * (exp(mu_small + 0.5 * sigma_small^2) - 1)
# # comp_disaster = lambda_disaster * mean(jump_disaster)
# comp_disaster = lambda_disaster * (exp(mu_disaster + 0.5 * sigma_disaster^2) - 1)
#
# for i_chunk = 1:num_chunks
#     print(string("Chunk ", i_chunk, " out of ", num_chunks, "\n"))
#
#     # Brownian component for volatility and price processes:
#     dW = zeros(len_path, 2, size_chunk)
#     for i_path = 1:size_chunk
#         dW[:,:, i_path] = rand(dW_normal, len_path)' .* sqrt(dt)
#     end
#
#     # Poisson processes to model jumps. First simulate poisson arrival proccess
#     N_small = zeros(len_path, size_chunk)
#     N_disaster = zeros(len_path, size_chunk)
#     for i_path = 1:size_chunk
#         N_small[:, i_path] = rand(poisson_small, len_path)
#         N_disaster[:, i_path] = rand(poisson_disaster, len_path)
#     end
#
#     # Now assign size of jumps to Poisson process:
#     for i_path = 1:size_chunk
#         num_jumps_small = length(N_small[N_small[:,i_path] .> 0, i_path])
#         if num_jumps_small > 0
#             N_small[N_small[:, i_path] .> 0, i_path] = rand(jump_small, num_jumps_small)
#         end
#
#         num_jumps_disaster = length(N_disaster[N_disaster[:,i_path] .> 0, i_path])
#         if num_jumps_disaster > 0
#             N_disaster[N_disaster[:, i_path] .> 0, i_path] = rand(jump_disaster, num_jumps_disaster)
#         end
#     end
#
#     # Calculating option prices for each path:
#     VS = zeros(len_path, 2, size_chunk)
#
#     # 1. Initializing first values:
#     VS[1,1,:] = ones(size_chunk) .* theta
#     VS[1,2,:] = ones(size_chunk) .* spot
#
#     for i_t = 2:len_path
#         V_prev = VS[i_t - 1, 1, :]
#         dW_V = dW[i_t, 1, :]
#
#         V_new = V_prev .+ k .* (theta .- V_prev) .* dt .+ sigma .* sqrt.(V_prev) .* dW_V  # updating volatility
#         V_new = max.(0.0, V_new)
#
#         S_prev = VS[i_t - 1, 2, :]
#         dW_S = dW[i_t, 2, :]
#         S_new = S_prev .+ S_prev .* r .* dt + sqrt.(V_new) .* S_prev .* dW_S  # updating stock price
#
#         # Doing jumps
#         # S_new = S_new .+ S_new .* N_small[i_t, :] .- S_new .* comp_small .* dt
#         # S_new = S_new .+ S_new .* N_disaster[i_t, :] .- S_new .* comp_disaster .* dt
#
#         S_new = S_new .+ S_new .* (exp.(N_small[i_t, :]) .- 1) .- S_new .* comp_small .* dt
#         S_new = S_new .+ S_new .* (exp.(N_disaster[i_t, :]) .- 1) .- S_new .* comp_disaster .* dt
#
#         VS[i_t, 1, :] = V_new
#         VS[i_t, 2, :] = S_new
#     end
#
#     # Saving the last price:
#     ind_start = (i_chunk - 1) * size_chunk + 1
#     ind_end = i_chunk * size_chunk
#     S1[ind_start:ind_end] = VS[end,2,:]
# end
#
#
# # Distribution of outcomes:
# Plots.histogram(S1)
#
# # Plots.plot(VS[:, 2, 100])
#
# # Plotting the distribution of shocks:
# draws_jump_small = rand(jump_small, 10000)
# draws_jump_disaster = rand(jump_disaster, 10000)
#
# Plots.histogram(exp.(draws_jump_small) .- 1, alpha = 0.5)
# Plots.histogram!(exp.(draws_jump_disaster) .- 1, alpha = 0.5)
#
#
#
# # i_path = 101
# # Plots.plot(LinRange(0,1,len_path), VS[:,2,i_path])
# # # Adding jumps to the plot:
# # time_jump = findall(x -> x .!= 0, N_small[:, i_path]) * dt
# # Plots.vline!(time_jump)
#
# ########################################################
# # Valuing options on different strikes:
# # 1. Generating list of strikes
# min_strike = spot * 0.4
# max_strike = spot * 1.6
# num_strikes = 25
#
# strike_list = LinRange(min_strike, max_strike, num_strikes)
# put_value = zeros(num_strikes)
# call_value = zeros(num_strikes)
#
# # 2. Calculating mean payoff (since we are operating in a risk-neutral measure)
# for i_strike = 1:length(strike_list)
#     strike = strike_list[i_strike]
#     put_value[i_strike] = exp(-r*T) * mean(max.(0, strike .- S1))
#     call_value[i_strike] = exp(-r*T) * mean(max.(0, S1 .- strike))
# end
#
# Plots.scatter(strike_list, put_value, label = "P(K)")
# Plots.scatter!(strike_list, call_value, label = "C(K)")
#
# ########################################################
# # Calculating BS implied volatility:
#
# # 1. BS call an dput prices:
# function BS_call_price(S0, q, r, K, sigma, T)
#     d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma * sqrt(T))
#     d2 = d1 - sigma*sqrt(T)
#
#     p1 = exp(-q*T) * S0 * cdf.(Normal(), d1)
#     p2 = exp(-r*T) * K * cdf.(Normal(), d2)
#
#     return p1 - p2
# end
#
# function BS_put_price(S0, q, r, K, sigma, T)
#     d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma * sqrt(T))
#     d2 = d1 - sigma*sqrt(T)
#
#     p1 = cdf.(Normal(), -d2) * K * exp(-r*T)
#     p2 = cdf.(Normal(), -d1) * S0 * exp(-q*T)
#
#     return p1 - p2
# end
#
# # 2. Calculating BS implied vol for each option:
# put_impl_vol = zeros(num_strikes)
# call_impl_vol = zeros(num_strikes)
#
# for i_strike = 1:num_strikes
#     function f!(F, x)
#         F[1] = BS_put_price(spot, 0, r, strike_list[i_strike], x[1], T) - put_value[i_strike]
#     end
#
#     put_impl_vol[i_strike] =  nlsolve(f!, [0.2]).zero[1]
# end
#
# for i_strike = 1:num_strikes
#     function f!(F, x)
#         F[1] = BS_call_price(spot, 0, r, strike_list[i_strike], x[1], T) - call_value[i_strike]
#     end
#
#     call_impl_vol[i_strike] = nlsolve(f!, [0.2]).zero[1]
# end
#
# Plots.scatter(strike_list, put_impl_vol, label = "from Puts", title = "Implied Volatilities")
# Plots.scatter!(strike_list, call_impl_vol, label = "from Calls")
#
# # 3. Now combine impled vols below and above the spot
# put_impl_vol_below_spot = put_impl_vol[strike_list .<= spot * exp(r*T)]
# call_impl_vol_above_spot = call_impl_vol[strike_list .> spot * exp(r*T)]
# impl_vol = vcat(put_impl_vol_below_spot, call_impl_vol_above_spot)
#
# Plots.scatter(log.(strike_list./spot), impl_vol.^2)
#
#
#
# ########################################################
# # Fitting SVI into the simulated volatility smile:
# # 1. Constructing option from simulated smile
#
# option = OptionData(1234, Dates.Date("1996-02-02"), Dates.Date("1997-02-02"),
#                     spot, strike_list[.!isequal.(NaN, impl_vol)], impl_vol[.!isequal.(NaN, impl_vol)], T, r, spot * exp(r*T))
#
# svi = fit_svi_zero_rho_global(option)
#
# # 2. Plotting fit:
# log_moneyness = log.(option.strikes/option.spot)
# impl_var = option.impl_vol
#
# range_log_moneyness = log_moneyness[end] - log_moneyness[1]
# plot_range = LinRange(log_moneyness[1] - range_log_moneyness*0.10,
#                       log_moneyness[end] + range_log_moneyness*0.10, 1000);
#
# Plots.scatter(log_moneyness, impl_vol.^2)
# Plots.plot!(plot_range, svi_smile(plot_range, svi.m,
#                                   svi.sigma, svi.rho,
#                                   svi.a, svi.b))
#
# ########################################################
# # Estimating different jump parameters:
# function calc_NTM_sigma(option::OptionData)
#     NTM_dist = 0.05
#     NTM_index = (option.strikes .<= option.spot*(1.0 + NTM_dist)) .&
#                 (option.strikes .>= option.spot*(1.0 - NTM_dist))
#     if sum(NTM_index) == 0
#         if minimum(option.strikes) > option.spot
#             sigma_NTM = option.impl_vol[1]
#         elseif maximum(option.strikes) < option.spot
#             sigma_NTM = option.impl_vol[end]
#         else
#             sigma_NTM = option.impl_vol[option.strikes .<= option.spot][end]
#         end
#     else
#         sigma_NTM = mean(option.impl_vol[NTM_index]) * sqrt(option.T)
#     end
#     return sigma_NTM
# end
#
# ests = estimate_parameters_mix(option.spot, option.int_rate, option.forward,
#         option.T, calc_NTM_sigma(option), minimum(option.strikes),
#         maximum(option.strikes), svi)
#
#
# D                   = ests[1] - ests[2]
# D_in_sample         = ests[3] - ests[4]
# D_bound             = ests[5] - ests[6]
# D_puts              = ests[7] - ests[8]
# D_bound_puts        = ests[9] - ests[10]
# D_deep_puts         = ests[11] - ests[12]
# D_bound_deep_puts   = ests[13] - ests[14]
# D_clamp             = ests[15] - ests[16]
# D_clamp_puts        = ests[17] - ests[18]
# D_clamp_deep_puts   = ests[19] - ests[20]
#
#
#
# # Plotting distribution of jumps:
# jump_small = Normal(-0.05, 0.05)
# jump_disaster = Normal(-0.6, 0.1)
# draws_jump_small = rand(jump_small, 10000)
# draws_jump_disaster = rand(jump_disaster, 10000)
#
# Plots.histogram(exp.(draws_jump_small) .- 1, alpha = 0.5, label = "Small Jumps", size = (600,400))
# Plots.histogram!(exp.(draws_jump_disaster) .- 1, alpha = 0.5, label = "Disaster Jumps")
# Plots.savefig("write-up-files/Simulations/images/jump_distribution_alt.pdf")
#
# ########################################################
# # Doing simulations in a compact form:
