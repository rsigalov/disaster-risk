print("\n ---- Loading libraries ----\n")

using DataFrames
using Dierckx # Package for interpolation
using Statistics
using LinearAlgebra

using CSV
using Dates

cd("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

days = ARGS[1]  # 1st argument: interpolation days
var_to_calculate = Symbol(ARGS[2]) # 2nd argument: variable to estimate
truncation_level = tryparse(Float64, ARGS[3]) # 3rd argument:
obs_filter = 15
share_months = 0.8
print("\n--- Loading Data ----\n")


# 1. reading files from the directory
dir_path = "estimated_data/interpolated_D/"
file_to_load = string("int_D_clamp_days_", days, ".csv")
# file_to_load = string("int_D_clamp_days_30.csv")

df = CSV.read(string(dir_path, file_to_load); datarow = 2, delim = ",")
df[:date_adj] = Dates.lastdayofmonth.(df[:date])

function filter_calc_mean(v, level)
    low_quant = quantile(v, level)
    upp_quant = quantile(v, 1-level)
    return mean(v[(v .>= low_quant) .& (v .<= upp_quant)])
end

function aggregate_disaster_measure(df, var_to_calculate, trunc)
    # Calculating cross-sectional average with truncation at 1% and 99% levels
    # for each month
    D_average_all = by(
        df[.!isequal.(df[var_to_calculate], NaN) .&
           .!isequal.(df[var_to_calculate], Inf) .&
           .!isequal.(df[var_to_calculate], -Inf) .&
           .!isequal.(df[var_to_calculate], missing), :],
        :date_adj,
        D = [var_to_calculate] => x -> filter_calc_mean(x[var_to_calculate], trunc))

    sort!(D_average_all, :date_adj)

    # Leaving only secid-month's where there is at least 15 days of observations:
    df_filter_days = by(
        df[.!isequal.(df[var_to_calculate], NaN) .&
           .!isequal.(df[var_to_calculate], Inf) .&
           .!isequal.(df[var_to_calculate], -Inf), :],
           [:secid, :date_adj],
           N = :secid => length,
           mean_all = var_to_calculate => x -> filter_calc_mean(x, 0.01))

    df_filter_days = df_filter_days[df_filter_days.N .>= obs_filter, :]

    # leaving companies that have >= 15 observations in at least 80% of the sample
    total_num_months = length(unique(df_filter_days[:date_adj]))
    num_months_for_secid = by(
        df_filter_days, :secid, N = :date_adj => x -> length(unique(x))
    )
    # Forming list of companies that satisfy the requirement:
    secid_list = num_months_for_secid[num_months_for_secid[:N] .>= total_num_months * share_months,:secid]

    # Leaving secid-months that satisfy the requirement
    df_filter_secid = df_filter_days[map(x -> x in secid_list, df_filter_days[:secid]), :] # Weird way to do the list version of <in> command
    df_filter_secid = df_filter_secid[:, [:secid, :date_adj, :mean_all]]

    # Pivoting the dataframe with D:
    df_pivot = unstack(df_filter_secid, :date_adj, :secid, :mean_all)

    # Replacing missing values with average D for each column:
    N_secid = size(df_pivot)[2] - 1
    for i_col = 2:(N_secid+1)
        mean_col = mean(skipmissing(df_pivot[:,i_col]))
        df_pivot[ismissing.(df_pivot[:,i_col]), i_col] = mean_col
    end

    # Converting dataframe to array
    date_vec = df_pivot[:date_adj]
    X_D = convert(Array{Float64,2}, df_pivot[:, 2:(N_secid+1)])

    # Doing PCA:
    cor_X = cor(X_D) # Calculating correlation matrix of D for secids
                     # to do Eigenvector decomposition on it

    # Getting eigenvectors and eigenvalues
    eigen_values, eigen_vectors = eigen(cor_X)

    # Finding the largest eigenvalue and getting the corresponding eigenvector...
    i_max_eigval = findmax(eigen_values)[2]

    # ... that corresponds to weights of different secids:
    w = eigen_vectors[:, i_max_eigval]
    w = w/sum(w)

    # Claculating first pricncipal component:
    Dw = X_D * w

    # Plotting first PCA
    # Plots.plot(date_vec, Dw)

    # Comparing PC1 with mean (across the same SECIDs):
    D_mean_filter = mean(X_D; dims = 2)

    # Plots.plot(date_vec, Dw, label = "PC1", title = "clamped D")
    # Plots.plot!(date_vec, D_mean_filter, label = "mean D (filter)")
    # Plots.plot!(date_vec, D_average_all.D, label = "mean D (all)")

    # Saving monthly series of PCA and mean D
    df_to_output = DataFrame(
        date = date_vec[:],
        pc1 = Dw,
        mean_filter = D_mean_filter[:],
        mean_all = D_average_all.D)

    # renaming columns according to the variable used:
    new_names = Symbol.(["date", string(var_to_calculate,"_pc1"),
        string(var_to_calculate, "_mean_filter"), string(var_to_calculate,"_mean_all")])
    names!(df_to_output, new_names)

    return df_to_output
end

df_agg = aggregate_disaster_measure(df, var_to_calculate, truncation_level)

CSV.write(string("estimated_data/disaster-risk-series/agg_", ARGS[2], "_", ARGS[1], "days.csv"), df_agg)

# eligible_columns = [:D_clamp, :rn_prob_2sigma, :rn_prob_20mon, :rn_prob_40mon,
#     :rn_prob_5mon, :rn_prob_10mon, :rn_prob_15mon]
#
# for column in names(df)
#
# end
#
# df_agg = aggregate_disaster_measure(df, :rn_prob_20mon, 0.0)
# df_agg = aggregate_disaster_measure(df, :D_clamp, 0.0)
# df_agg = aggregate_disaster_measure(df, :rn_prob_2sigma, 0.0)
