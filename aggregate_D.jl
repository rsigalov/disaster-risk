print("\n ---- Loading libraries ----\n")

using DataFrames
using Dierckx # Package for interpolation
using Statistics
using LinearAlgebra

using CSV
using Dates

cd("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

days = ARGS[1]  # 1st argument: interpolation days
# var_to_calculate = Symbol(ARGS[2]) # 2nd argument: variable to estimate
# truncation_level = tryparse(Float64, ARGS[2]) # 3rd argument:
obs_filter = 10
truncation_level = 0.01

print("\n---- Loading Data ----\n")

# 1. reading files from the directory
dir_path = "estimated_data/interpolated_D/"
file_to_load = string("int_ind_disaster_days_", days, ".csv")
# file_to_load = string("int_ind_disaster_days_120.csv")

df = CSV.read(string(dir_path, file_to_load); datarow = 2, delim = ",")
df[:date_adj] = Dates.lastdayofmonth.(df[:date])

function filter_calc_mean(v, level)
    low_quant = quantile(v, level)
    upp_quant = quantile(v, 1-level)
    return mean(v[(v .>= low_quant) .& (v .<= upp_quant)])
end

function aggregate_disaster_measure(df, var_to_calculate)

    # Calculating cross-sectional average with truncation at 1% and 99% levels
    # for each month. Removing NaN, inf and missing values
    D_average_all = by(
        df[.!isequal.(df[var_to_calculate], NaN) .&
           .!isequal.(df[var_to_calculate], Inf) .&
           .!isequal.(df[var_to_calculate], -Inf) .&
           .!isequal.(df[var_to_calculate], missing), :],
        :date_adj,
        D = [var_to_calculate] => x -> filter_calc_mean(x[var_to_calculate], truncation_level))

    sort!(D_average_all, :date_adj)

    # Leaving only secid-month's where there is at least 10 days of observations:
    df_filter_days = by(
        df[.!isequal.(df[var_to_calculate], NaN) .&
           .!isequal.(df[var_to_calculate], Inf) .&
           .!isequal.(df[var_to_calculate], -Inf) .&
           .!isequal.(df[var_to_calculate], missing), :],
           [:secid, :date_adj],
           N = :secid => length,
           mean_all = var_to_calculate => x -> filter_calc_mean(x, truncation_level))

    df_filter_days = df_filter_days[df_filter_days.N .>= obs_filter, :]

    D_filter_days = by(
        df_filter_days, :date_adj, D = :mean_all => x -> filter_calc_mean(x, truncation_level)
    )

    sort!(D_filter_days, :date_adj)

    ############################################################################
    # Commented PC1 since it requires to have a relatively full panel which
    # is not feasible when I use only smiles that straddle 30 days
    ############################################################################

    # # leaving companies that have >= 15 observations in at least 80% of the sample
    # total_num_months = length(unique(df_filter_days[:date_adj]))
    # num_months_for_secid = by(
    #     df_filter_days, :secid, N = :date_adj => x -> length(unique(x))
    # )
    # # Forming list of companies that satisfy the requirement:
    # secid_list = num_months_for_secid[num_months_for_secid[:N] .>= total_num_months * share_months,:secid]
    #
    # # Leaving secid-months that satisfy the requirement
    # df_filter_secid = df_filter_days[map(x -> x in secid_list, df_filter_days[:secid]), :] # Weird way to do the list version of <in> command
    # df_filter_secid = df_filter_secid[:, [:secid, :date_adj, :mean_all]]
    #
    # # Pivoting the dataframe with D:
    # df_pivot = unstack(df_filter_secid, :date_adj, :secid, :mean_all)
    #
    # # Replacing missing values with average D for each column:
    # N_secid = size(df_pivot)[2] - 1
    # for i_col = 2:(N_secid+1)
    #     mean_col = mean(skipmissing(df_pivot[:,i_col]))
    #     df_pivot[ismissing.(df_pivot[:,i_col]), i_col] = mean_col
    # end
    #
    # # Converting dataframe to array
    # date_vec = df_pivot[:date_adj]
    # print(size(date_vec))
    # X_D = convert(Array{Float64,2}, df_pivot[:, 2:(N_secid+1)])
    #
    # # Doing PCA:
    # cor_X = cor(X_D) # Calculating correlation matrix of D for secids
    #                  # to do Eigenvector decomposition on it
    #
    # # Getting eigenvectors and eigenvalues
    # eigen_values, eigen_vectors = eigen(cor_X)
    #
    # # Finding the largest eigenvalue and getting the corresponding eigenvector...
    # i_max_eigval = findmax(eigen_values)[2]
    #
    # # ... that corresponds to weights of different secids:
    # w = eigen_vectors[:, i_max_eigval]
    # w = w/sum(w)
    #
    # # Claculating first pricncipal component:
    # Dw = X_D * w
    #
    # # Comparing PC1 with mean (across the same SECIDs):
    # D_mean_filter = mean(X_D; dims = 2)
    #
    # # Saving monthly series of PCA and mean D
    # df_to_output = DataFrame(
    #     date = date_vec[:],
    #     pc1 = Dw,
    #     mean_filter_days = D_filter_days.D,
    #     mean_filter = D_mean_filter[:],
    #     mean_all = D_average_all.D)

    # df_to_output = DataFrame(
    #     date = D_average_all.date_adj,
    #     mean_all = D_average_all.D,
    #     mean_filter_days = D_filter_days.D
    #     )

    # returning two dataframes:
    df_to_output_1 = DataFrame(
        date = D_average_all.date_adj,
        value = D_average_all.D,
        var = string(var_to_calculate),
        agg_type = "mean_all"
    )
    df_to_output_2 = DataFrame(
        date = D_average_all.date_adj,
        value = D_filter_days.D,
        var = string(var_to_calculate),
        agg_type = "mean_filter"
    )

    # renaming columns according to the variable used:
    # new_names = Symbol.(["date", string(var_to_calculate,"_pc1"),
    #     string(var_to_calculate, "_mean_filter"), string(var_to_calculate,"_mean_all")])

    return df_to_output_1, df_to_output_2
end

column_to_interpolate = names(df)[3:end]
column_to_interpolate = column_to_interpolate[column_to_interpolate .!= :D]
column_to_interpolate = column_to_interpolate[column_to_interpolate .!= :date_adj]

i_col = 0

print("\n ---- Starting Aggregation ----\n")

df_agg = DataFrame(
    date = [Dates.Date("1996-02-02")], value = [1.0], var = ["M"], agg_type = ["M"])

for var_to_calculate in column_to_interpolate
    @show var_to_calculate
    df_agg_1, df_agg_2 = aggregate_disaster_measure(df, var_to_calculate)
    append!(df_agg, df_agg_1)
    append!(df_agg, df_agg_2)
end

df_agg[:days] = days
df_agg[:level] = "ind"
df_agg = df_agg[2:end,:]

print("\n ---- Saving Results ----\n")

CSV.write(string("estimated_data/disaster-risk-series/agg_combined_", ARGS[1], "days.csv"), df_agg)

print("\n ---- Done ----\n")
