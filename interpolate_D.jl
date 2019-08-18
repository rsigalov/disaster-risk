############################################################
# This script does the calculation of D for
# given maturity (say 30 days) and a given measure
# (say D-in-sample).
############################################################

print("\n ---- Loading libraries ----\n")

using DataFrames
using Dierckx # Package for interpolation

using CSV
using Dates

days = tryparse(Float64, ARGS[1])


print("\n--- Loading Data ----\n")

# 1. Reading files from the directory with disaster D-measure and monthly
# (standardized decline probabilities).
dir_path = "estimated_data/V_IV/"
file_list = readdir(dir_path)
file_list = file_list[map(x -> occursin("var_ests_final", x), file_list)]

df = CSV.read(string(dir_path, file_list[1]); datarow = 2, delim = ",")

for i in 1:length(file_list)
    if i == 1
        global df = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
    else
        df_append = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
        # global df = vcat(df, df_append)
        append!(df, df_append)
    end
end

rename!(df, Dict(:rn_prob_2sigma => :rn_prob_2sigma_ann))

# 2. Getting data on non-standardized risk neutral probability of a decline.
dir_path = "estimated_data/rn_prob_new/"
file_list = readdir(dir_path)
file_list = file_list[map(x -> occursin("var_ests_final", x), file_list)]

df_rn_prob = CSV.read(string(dir_path, file_list[1]); datarow = 2, delim = ",")

for i in 1:length(file_list)
    if i == 1
        global df_rn_prob = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
    else
        df_append = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
        global df_append = vcat(df_append, df_append)
    end
end

# 3. Calculating Ds and droppping the rest of the variables:
df[:D] = df[:V] - df[:IV]
df[:D_clamp] = df[:V_clamp] - df[:IV_clamp]
df[:D_in_sample] = df[:V_in_sample] - df[:IV_in_sample]

df = df[:,[:secid, :date, :T, :D, :D_clamp, :D_in_sample, :rn_prob_2sigma_ann, :rn_prob_20ann, :rn_prob_40ann]]

# 2. Writing function to calculate measure given dataframe, measure name and length:
function calculate_D(sub_df, measure, days)
    # Setting up coordinates:
    x = sub_df[:T]
    y = sub_df[measure]

    # picking non-missing coordinates:
    y_non_missing = .!(isequal.(y, NaN) .| isequal(y, Inf)  .| isequal.(y, missing))
    x = x[y_non_missing]
    y = y[y_non_missing]

    # Using only straddled maturities:
    y_to_interp = days/365
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
end

sort!(df, [:secid, :date, :T])

df_grouped = groupby(df, [:secid, :date])

print("\n--- First Pass ----\n")
@time tmp = map(x -> calculate_D(x, :D_clamp, days), [i for i in df_grouped][1:2])
print("\n--- Second Pass ----\n")
@profile var_arr = map(x -> calculate_D(x, :D_clamp, days), [i for i in df_grouped[1:1000]])

# First interpolate previosuly calculated variables: D's and adjusted risk
# neutral probabilities
var_list = [:D, :D_in_sample, :D_clamp, :rn_prob_2sigma_ann, :rn_prob_20ann, :rn_prob_40ann]

@time var_arr = map(x -> calculate_D(x, :D_clamp, days), [i for i in df_grouped[1:100]])

df_grouped = groupby(
    df[.!(isequal.(df[:D_clamp], NaN) .| isequal.(df[:D_clamp], Inf)  .| isequal.(df[:D_clamp], missing)),[:secid,:date,:T,:D_clamp]],
    [:secid, :date])

var_arr = Array{Tuple{Int, Date, Float64}}(undef, 10000)
i = 0
secid_arr = zeros(10000)
date_arr = zeros(10000)
D_arr = zeros(10000)
@time for sub_df in df_grouped[1:10000]
    global i += 1
    var_arr[i] = calculate_D(sub_df, :D_clamp, days)
end

# What if I pass the vectors instead of sub_df





for j in 1:length(var_list)
    var = var_list[j]
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

# Next, interpolate risk neutral probabilities for fixed thresholds, linearly
# as a first pass.
df_grouped = groupby(df_rn_prob, [:secid, :date])
var_list = [:rn_prob_sigma, :rn_prob_2sigma, :rn_prob_20, :rn_prob_40, :rn_prob_60, :rn_prob_80]

for j in 1:length(var_list)
    var = var_list[j]
    var_arr = map(x -> calculate_D(x, var, days), [i for i in df_grouped])
    df_to_join = DataFrame(
        secid = map(x -> x[1], var_arr),
        date = map(x -> x[2], var_arr),
        D = map(x -> x[3], var_arr))
    rename!(df_to_join, Dict(:D => var))
    global df_to_save = join(df_to_save, df_to_join, on = [:secid, :date])
end

print("\n--- Outputting Data ----\n")
CSV.write(string("estimated_data/interpolated_D/int_ind_disaster_days_", ARGS[1], ".csv"), df_to_save)
