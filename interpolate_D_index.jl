############################################################
# This script does the calculation of D for
# given maturity (say 30 days) and a given measure
# (say D-in-sample)
############################################################

print("\n ---- Loading libraries ----\n")

using DataFrames
using Dierckx # Package for interpolation
using Statistics

using CSV
using Dates

index_to_append = ARGS[1]
days = tryparse(Float64, ARGS[2])

print("\n--- Loading Data ----\n")

# 1. reading files from the directory
dir_path = "estimated_data/V_IV/"
file_list = readdir(dir_path)

df = CSV.read(string(dir_path, string("var_ests_", index_to_append, ".csv")))

# 2. Calculating Ds and droppping the rest of the variables:
df[:D] = df[:V] - df[:IV]
df[:D_clamp] = df[:V_clamp] - df[:IV_clamp]
df[:D_in_sample] = df[:V_in_sample] - df[:IV_in_sample]

# 3. Writing function to calculate measure given dataframe, measure name and length:
function calculate_D(sub_df, measure, days)
    # Setting up coordinates:
    x = sub_df[:T]
    y = sub_df[measure]

    # picking non-missing coordinates:
    y_non_missing = .!(isequal.(y, NaN) .| isequal(y, Inf)  .| isequal.(y, missing))
    x = x[y_non_missing]
    y = y[y_non_missing]

    # Using only straddled maturities:
    y_to_interp = (days - 1)/365
    if length(x) == 0
        to_return =  NaN
    elseif length(x) == 1
        if x == y_to_interp
            to_return =  y
        else
            to_return =  NaN
        end
    elseif (minimum(x) <= y_to_interp) & (maximum(x) >= y_to_interp)
        interp_rate = Spline1D(x, y, k = 1)
        to_return = interp_rate(y_to_interp)
    else
        to_return = NaN
    end

    # returning interpolated D along with keys: secid, date
    return (sub_df.secid[1], sub_df.date[1], to_return)
    # return to_return
end

df = sort(df, cols = [:date, :T])

df_grouped = groupby(df, :date)

print("\n--- First Pass ----\n")
@time tmp = map(x -> calculate_D(x, :D_clamp, days), [i for i in df_grouped][1:2])
print("\n--- Second Pass ----\n")

var_list = [:D, :D_in_sample, :D_clamp, :rn_prob_sigma, :rn_prob_2sigma,
    :rn_prob_5, :rn_prob_10, :rn_prob_15, :rn_prob_20]

for j in 1:length(var_list)
    var = var_list[j]
    @show var
    if j == 1
        var_arr = map(x -> calculate_D(x, var, days), [i for i in df_grouped])
        global df_to_save = DataFrame(
            secid = map(x -> x[1], var_arr),
            date = map(x -> x[2], var_arr),
            D = map(x -> x[3], var_arr))
        rename!(df_to_save, Dict(:D => var))
    else
        var_arr = map(x -> calculate_D(x, var, days), [i for i in df_grouped])
        df_to_join = DataFrame(
            secid = map(x -> x[1], var_arr),
            date = map(x -> x[2], var_arr),
            D = map(x -> x[3], var_arr))
        rename!(df_to_join, Dict(:D => var))
        global df_to_save = join(df_to_save, df_to_join, on = [:secid, :date])
    end
end

print("\n--- Outputting Data ----\n")
CSV.write(string("data/interpolated_D/int_D_", index_to_append, "_days_", ARGS[2], ".csv"), df_to_save)



# # Loading all data:
# df = CSV.read("estimated_data/disaster-risk-series/int_D_spx_days_60.csv")
# df = df[:, [:date, :D_clamp]]
# df2 = CSV.read("estimated_data/disaster-risk-series/int_D_spx_old_CME_days_60.csv")
# df2 = df2[:, [:date, :D_clamp]]
#
# append!(df, df2)
# df[:date_mon] = Dates.lastdayofmonth.(df[:date])
# av_D = by(df, :date_mon, av_D = :D_clamp => mean)
#
# Plots.plot(av_D.date_mon, av_D.av_D)
#
#
#
# min_max_T = by(df, :date, min_T = :T => minimum, max_T = :T => maximum)
# Plots.plot(min_max_T.date, min_max_T.min_T.*365, label = "Minimum maturity", legend = :topleft)
# Plots.plot!(min_max_T.date, min_max_T.max_T.*365, label = "Maxmimum maturity")
# Plots.hline!([30, 40, 60, 120, 180], label = "30, 40, 60, 120, 180 days")
