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

CSV.write(string("estimated_data/simulations/simulation_results_prob_case_", case, ".csv"), df_out_prob)


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

########################################################################
# Plotting distribution of terminal price of the stock for different
# cases for the Jump process.
########################################################################

# Estimating different cases
comp_sims = map(x -> do_simulation(x, 0, -0.05, 0.05)[1], [0, 5, 20, 40])


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
