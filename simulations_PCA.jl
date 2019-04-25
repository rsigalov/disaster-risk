# cd("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

using DataFrames
using LinearAlgebra # Package with some useful functions
using Distributions # Package for normal CDF
using IterTools

using CSV
using Plots

normal = Normal(0, 1) # random variable generator for all noraml innovations
lognormal = LogNormal(0, 0.5) # Random variable generator for loadings

# Function to simulate AR(1) with correct initial draws:
function generate_AR_1(T, N, phi, std_e) # phi is AR1, phi = sqrt(rho) in the problem set
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

# ar1 = generate_AR_1(264, 180, 0.9, 1)

function simulate_many_pcs(rho_d, rho_i, std_e_i, n_sims, n_firms)

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

corr_pc1_arr, corr_mean_arr = simulate_many_pcs(0.9, 0.9, 4, 1000, 30)
Plots.histogram(abs.(corr_pc1_arr))
Plots.histogram!(abs.(corr_mean_arr))

std_e_i_list = [2, 4, 8]
rho_d_list = [0.25, 0.5, 0.9]
rho_i_list = [0.05, 0.5, 0.9]
num_firms_list = [300, 500, 750, 1000]

df_parameters = DataFrame(std_e_i = [1.0, 2.0], rho_d = [1.0, 2.0],
                   rho_i = [1.0, 2.0], num_firms = [1, 2])

# Matrix of parameters:
for p in Iterators.product(std_e_i_list,rho_d_list,rho_i_list,num_firms_list)
    df_to_append = DataFrame(std_e_i = p[1], rho_d = p[2], rho_i = p[3], num_firms = p[4])
    append!(df_parameters, df_to_append)
end

df_parameters = df_parameters[3:end,:]

# Simulating PCA with all parameters and writing into a data frame:
df_sim_results = DataFrame(std_e_i = [1.0, 2.0], rho_d = [1.0, 2.0],
                          rho_i = [1.0, 2.0], num_firms = [1.0, 2.0],
                          est = ["A", "B"], mean_est = [1.0, 2.0],
                          med_est = [1.0,2.0], quant_5 = [1.0, 2.0],
                          quant_95 = [1.0, 2.0])

for i_row = 1:size(df_parameters)[1]
# for i_row = 1:2
    @show i_row
    rho_d = df_parameters[i_row, 2]
    rho_i = df_parameters[i_row, 3]
    std_e_i = df_parameters[i_row, 1]
    num_firms = df_parameters[i_row, 4]

    # 1. Simulating PCA
    corr_pc1_arr, corr_mean_arr =
        simulate_many_pcs(rho_d, rho_i, std_e_i, 1000, num_firms)

    # 2. Calculating Summary statistics for PC1
    mean_pc1 = mean(abs.(corr_pc1_arr))
    median_pc1 = median(abs.(corr_pc1_arr))
    quant_5_pc1 = quantile(abs.(corr_pc1_arr), 0.05)
    quant_95_pc1 = quantile(abs.(corr_pc1_arr), 0.95)

    df_to_append = DataFrame(std_e_i = std_e_i, rho_d = rho_d, rho_i = rho_i,
        num_firms = num_firms, est = "PC1", mean_est = mean_pc1,
        med_est = median_pc1, quant_5 = quant_5_pc1, quant_95 = quant_95_pc1)
    append!(df_sim_results, df_to_append)

    # 3. Caculating summary statistics for meanD
    mean_meanD = mean(abs.(corr_mean_arr))
    median_meadD = median(abs.(corr_mean_arr))
    quant_5_meadD = quantile(abs.(corr_mean_arr), 0.05)
    quant_95_meanD = quantile(abs.(corr_mean_arr), 0.95)

    df_to_append = DataFrame(std_e_i = std_e_i, rho_d = rho_d, rho_i = rho_i,
        num_firms = num_firms, est = "meanD", mean_est = mean_meanD,
        med_est = median_meadD, quant_5 = quant_5_meadD, quant_95 = quant_95_meanD)
    append!(df_sim_results, df_to_append)

end

df_sim_results = df_sim_results[3:end, :]

CSV.write("simulation_pca_summary_stats_2.csv", df_sim_results)


############################################
# Writing a function to do the same in a parallelized wa with pmap:



std_e_i_list = [2, 4, 8]
rho_d_list = [0.25, 0.5, 0.9]
rho_i_list = [0.05, 0.5, 0.9]
num_firms_list = [300, 500, 750, 1000]

[repmat(std_e_i_list,1,length(rho_d_list))'[:] repmat(rho_d_list,length(std_e_i_list),1)[:]]
