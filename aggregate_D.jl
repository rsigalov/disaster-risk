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
obs_filter = 5
truncation_level = 0.025

print("\n---- Loading Data ----\n")

# 1. reading files from the directory
dir_path = "estimated_data/interpolated_D/"
# file_to_load = string("int_ind_disaster_days_", days, ".csv")
file_to_load = string("int_ind_disaster_union_cs_", days, ".csv")
# file_to_load = string("int_ind_disaster_days_120.csv")

df = CSV.read(string(dir_path, file_to_load); datarow = 2, delim = ",")
df[:date_adj] = Dates.lastdayofmonth.(df[:date])

function filter_calc_mean(v, level)
    low_quant = quantile(v, level)
    upp_quant = quantile(v, 1 - level)
    return mean(v[(v .>= low_quant) .& (v .<= upp_quant)])
end

function aggregate_disaster_measure(df, var_to_calculate)

    # Calculating cross-sectional average with truncation at 1% and 99% levels
    # for each month. Removing NaN, inf and missing values
    df_mon_mean = by(df[.!isequal.(df[var_to_calculate], NaN) .&
           .!isequal.(df[var_to_calculate], Inf) .&
           .!isequal.(df[var_to_calculate], -Inf) .&
           .!isequal.(df[var_to_calculate], missing), :],
           [:secid, :date_adj],
           N = :secid => length,
           mean_all = var_to_calculate => mean)

    D_mean_all = by(
        df_mon_mean, :date_adj,
        D = :mean_all => mean,
        num_comps = :secid => length)

    df_filter = df_mon_mean[df_mon_mean.N .>= obs_filter, :]

    D_filter = by(
        df_filter, :date_adj,
        D = :mean_all => x -> filter_calc_mean(x, truncation_level),
        num_comps = :mean_all => length)

    sort!(D_mean_all, :date_adj)
    sort!(D_filter, :date_adj)

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
        date = D_mean_all.date_adj,
        agg_type = "mean_all",
        var = string(var_to_calculate),
        value = D_mean_all.D,
        num_comps = D_mean_all.num_comps
        )

    df_to_output_2 = DataFrame(
        date = D_filter.date_adj,
        var = string(var_to_calculate),
        agg_type = "mean_filter",
        value = D_filter.D,
        num_comps = D_filter.num_comps,
        )

    # renaming columns according to the variable used:
    # new_names = Symbol.(["date", string(var_to_calculate,"_pc1"),
    #     string(var_to_calculate, "_mean_filter"), string(var_to_calculate,"_mean_all")])

    return df_to_output_1, df_to_output_2
end

column_to_interpolate = names(df)[3:end]
column_to_interpolate = column_to_interpolate[column_to_interpolate .!= :D]
column_to_interpolate = column_to_interpolate[column_to_interpolate .!= :date_adj]
column_to_interpolate = column_to_interpolate[column_to_interpolate .!= :date]
column_to_interpolate = column_to_interpolate[column_to_interpolate .!= :crsp_cs]
column_to_interpolate = column_to_interpolate[column_to_interpolate .!= :issue_type]
column_to_interpolate = column_to_interpolate[column_to_interpolate .!= :old_obs]

i_col = 0

print("\n ---- Starting Aggregation ----\n")

global df_agg = DataFrame(
    date = [Dates.Date("1996-02-02")], var = ["M"], agg_type = ["M"], value = [1.0], num_comps = [1])

for var_to_calculate in column_to_interpolate
    @show var_to_calculate
    df_agg_1, df_agg_2 = aggregate_disaster_measure(df, var_to_calculate)
    global df_agg = vcat(df_agg, df_agg_1)
    global df_agg = vcat(df_agg, df_agg_2)
end

df_agg[:days] = days
df_agg[:level] = "ind"
df_agg = df_agg[2:end, :]

print("\n ---- Saving Results ----\n")

CSV.write(string("estimated_data/disaster-risk-series/agg_combined_union_cs_", ARGS[1], "days.csv"), df_agg)

print("\n ---- Done ----\n")
