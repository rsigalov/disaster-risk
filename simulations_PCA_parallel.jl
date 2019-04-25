# cd("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

@everywhere using Distributed
@everywhere using DataFrames
@everywhere using LinearAlgebra # Package with some useful functions
@everywhere using Distributions # Package for normal CDF

using CSV

@everywhere normal = Normal(0, 1) # random variable generator for all noraml innovations
@everywhere lognormal = LogNormal(0, 0.5) # Random variable generator for loadings
@everywhere T = 264

# Function to simulate AR(1) with correct initial draws:
@everywhere function generate_AR_1(T, N, phi, std_e) # phi is AR1, phi = sqrt(rho) in the problem set
    x = zeros(T,N)

    # Initializing AR(1) process:
    for n = 1:N
        nu = rand(normal, 2, 1)
        F = [phi 0; 0 0]
        V_p = inv(I - kron(F,F)) * ones(4,1)
        V = reshape(V_p, (2,2))
        CV = cholesky(V).U
        z = CV * nu
        x_1 = z[1]

        x[1, n] = x_1
    end

    # Generating subsequent random innovations to AR(1) process
    e = rand(normal, T - 1, N) * std_e

    # looping through periods 2,...,T to fill AR(1) process
    for n = 1:N
        for t = 2:T
            x[t, n] = phi * x[t-1, n] + e[t-1,n]
        end
    end

    return x
end

@everywhere function simulate_many_pcs(rho_d, rho_i, std_e_i, n_sims, n_firms)

    corr_pc1_list = zeros(n_sims)
    corr_mean_list=  zeros(n_sims)

    for i_sim = 1:n_sims

        # 1. Simulate AR(1) common component:
        L_d = generate_AR_1(T, 1, rho_d, 1)

        # 2. Picking random loadings:
        loadings = rand(lognormal, n_firms)

        # 3. Simulate AR(1) idiosyncratic components:
        L_i = generate_AR_1(T, n_firms, rho_i, std_e_i)

        # 4. Generating all time series:
        D_sim = L_d * loadings' + L_i

        # 5. Running PCA
        pc1 = eigvecs(D_sim * D_sim')[:, end]
        corr_pc1_list[i_sim] = cor(pc1, L_d)[1]

        # 6. Calculating cross-sectional mean as one of the approaches
        # to identify the common factor
        mean_D = mean(D_sim; dims = 2)
        corr_mean_list[i_sim] = cor(mean_D, L_d)[1]
    end

    return corr_pc1_list, corr_mean_list
end

std_e_i_list = [2, 4, 8]
rho_d_list = [0.25, 0.5, 0.9]
rho_i_list = [0.05, 0.5, 0.9]
num_firms_list = [300, 500, 750, 1000]
total_length = length(std_e_i_list) * length(rho_d_list) * length(rho_i_list) * length(num_firms_list)

std_e_i_arr = zeros(total_length)
rho_d_arr = zeros(total_length)
rho_i_arr = zeros(total_length)
num_firms_arr = zeros(total_length)

i = 0
for std_e_i in std_e_i_list
    for rho_d in rho_d_list
        for rho_i in rho_i_list
            for num_firms in num_firms_list
                global i += 1
                std_e_i_arr[i] = std_e_i
                rho_d_arr[i] = rho_d
                rho_i_arr[i] = rho_i
                num_firms_arr[i] = num_firms
            end
        end
    end
end

num_firms_arr = convert.(Int, num_firms_arr)

@everywhere struct sum_stat
    mean
    median
    quant_5
    quant_95
end

@everywhere function pc_summary_stats(std_e_i, rho_d, rho_i, num_firms)
    # 1. Simulating PCA
    corr_pc1_arr, corr_mean_arr =
        simulate_many_pcs(rho_d, rho_i, std_e_i, 1000, num_firms)

    # 2. Calculating Summary statistics for PC1
    mean_pc1 = mean(abs.(corr_pc1_arr))
    median_pc1 = median(abs.(corr_pc1_arr))
    quant_5_pc1 = quantile(abs.(corr_pc1_arr), 0.05)
    quant_95_pc1 = quantile(abs.(corr_pc1_arr), 0.95)

    pc1_sum_stat = sum_stat(mean_pc1,median_pc1,quant_5_pc1,quant_95_pc1)

    # 3. Caculating summary statistics for meanD
    mean_meanD = mean(abs.(corr_mean_arr))
    median_meadD = median(abs.(corr_mean_arr))
    quant_5_meadD = quantile(abs.(corr_mean_arr), 0.05)
    quant_95_meanD = quantile(abs.(corr_mean_arr), 0.95)

    mean_sum_stat = sum_stat(mean_meanD,median_meadD,quant_5_meadD,quant_95_meanD)

    return pc1_sum_stat, mean_sum_stat
end

print("\n--- Starting Simulations ----\n")
print("\n--- First Pass ----\n")
@time ests = pmap(pc_summary_stats, std_e_i_arr[1:2], rho_d_arr[1:2],
    rho_i_arr[1:2], num_firms_arr[1:2])
print("\n--- Second Pass ----\n")
@time ests = pmap(pc_summary_stats, std_e_i_arr, rho_d_arr, rho_i_arr, num_firms_arr)

print("\n--- Aggregating and Saving Results ----\n")
# Unpacking all the stuff and saving as CSV:
df_sim_results = DataFrame(std_e_i = [1.0, 2.0], rho_d = [1.0, 2.0],
                          rho_i = [1.0, 2.0], num_firms = [1.0, 2.0],
                          est = ["A", "B"], mean_est = [1.0, 2.0],
                          med_est = [1.0,2.0], quant_5 = [1.0, 2.0],
                          quant_95 = [1.0, 2.0])

for i = 1:length(ests)
    est = ests[i][1]
    df_to_append = DataFrame(std_e_i = std_e_i_arr[i], rho_d = rho_d_arr[i],
        rho_i = rho_i_arr[i], num_firms = num_firms_arr[i], est = "PC1",
        mean_est = est.mean, med_est = est.median, quant_5 = est.quant_5,
        quant_95 = est.quant_95)
    append!(df_sim_results, df_to_append)

    est = ests[i][2]
    df_to_append = DataFrame(std_e_i = std_e_i_arr[i], rho_d = rho_d_arr[i],
        rho_i = rho_i_arr[i], num_firms = num_firms_arr[i], est = "MeanD",
        mean_est = est.mean, med_est = est.median, quant_5 = est.quant_5,
        quant_95 = est.quant_95)
    append!(df_sim_results, df_to_append)
end

df_sim_results = df_sim_results[3:end,:]

CSV.write("simulation_pca_summary_stats_2.csv", df_sim_results)
