############################################################
# This script does the calculation of D for
# given maturity (say 30 days) and a given measure
# (say D-in-sample)
############################################################

print("\n ---- Loading libraries ----\n")

using DataFrames
using Dierckx # Package for interpolation

using CSV
using Dates

days = tryparse(Float64, ARGS[1])
# i_first = tryparse(Int, ARGS[2])
# i_last = tryparse(Int, ARGS[3])
# days = 30

print("\n--- Loading Data ----\n")

# 1. reading files from the directory
dir_path = "estimated_data/V_IV/"
file_list = readdir(dir_path)
# file_list = file_list[i_first:i_last]

for i in 1:length(file_list)
    if i == 1
        global df = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
    else
        df_append = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
        append!(df, df_append)
    end
end

# 2. Calculating Ds and droppping the rest of the variables:
df[:D] = df[:V] - df[:IV]
df[:D_clamp] = df[:V_clamp] - df[:IV_clamp]
df[:D_in_sample] = df[:V_in_sample] - df[:IV_in_sample]

df = df[:,[:secid, :date, :T, :D, :D_clamp, :D_in_sample, :rn_prob_2sigma, :rn_prob_20ann, :rn_prob_40ann]]

# 2. Writing function to calculate measure given dataframe, measure name and length:
function calculate_D(sub_df, measure, days)
    if size(sub_df[1])[1] == 1
        to_return = sub_df[measure][1]
    else
        # x and y for interpolation:
        x = sub_df[:T]
        y = sub_df[measure]
        # picking non-missing coordinates:
        y_non_missing = .!(isequal.(y, NaN) .| isequal(y, Inf))
        x = x[y_non_missing]
        y = y[y_non_missing]
        if length(x) == 1
            to_return = y[1]
        elseif length(x) == 0
            to_return = NaN
        else
            # interpolating:
            interp_rate = Spline1D(x, y, k = 1) # creating linear interpolation object
                                                # that we can use later as well

            to_return = interp_rate(days/365)
        end
    end

    # returning interpolated D along with keys: secid, date
    return (sub_df.secid[1], sub_df.date[1], to_return)
    # return to_return
end

sort!(df, [:secid, :date, :T])

df_grouped = groupby(df, [:secid, :date])

print("\n--- First Pass ----\n")
@time tmp = map(x -> calculate_D(x, :D_clamp, days), [i for i in df_grouped][1:2])
print("\n--- Second Pass ----\n")
@time D_arr = map(x -> calculate_D(x, :D, days), [i for i in df_grouped])
@time D_in_sample_arr = map(x -> calculate_D(x, :D_in_sample, days), [i for i in df_grouped])
@time D_clamp_arr = map(x -> calculate_D(x, :D_clamp, days), [i for i in df_grouped])
@time rn_prob_2sigma_arr = map(x -> calculate_D(x, :rn_prob_2sigma, days), [i for i in df_grouped])
@time rn_prob_20mon_arr = map(x -> calculate_D(x, :rn_prob_20ann, days), [i for i in df_grouped])
@time rn_prob_40mon_arr = map(x -> calculate_D(x, :rn_prob_40ann, days), [i for i in df_grouped])

D_df = DataFrame(
    secid = map(x -> x[1], D_arr),
    date = map(x -> x[2], D_arr),
    D = map(x -> x[3], D_arr))

D_in_sample_df = DataFrame(
    secid = map(x -> x[1], D_in_sample_arr),
    date = map(x -> x[2], D_in_sample_arr),
    D_in_sample = map(x -> x[3], D_in_sample_arr))

D_clamp_df = DataFrame(
    secid = map(x -> x[1], D_clamp_arr),
    date = map(x -> x[2], D_clamp_arr),
    D_clamp = map(x -> x[3], D_clamp_arr))

rn_prob_2sigma_df = DataFrame(
    secid = map(x -> x[1], rn_prob_2sigma_arr),
    date = map(x -> x[2], rn_prob_2sigma_arr),
    rn_prob_2sigma = map(x -> x[3], rn_prob_2sigma_arr))

rn_prob_20mon_df = DataFrame(
    secid = map(x -> x[1], rn_prob_20mon_arr),
    date = map(x -> x[2], rn_prob_20mon_arr),
    rn_prob_20mon = map(x -> x[3], rn_prob_20mon_arr))

rn_prob_40mon_df = DataFrame(
    secid = map(x -> x[1], rn_prob_40mon_arr),
    date = map(x -> x[2], rn_prob_40mon_arr),
    rn_prob_40mon = map(x -> x[3], rn_prob_40mon_arr))


# Merging all tables:
D_to_save = join(D_df, D_in_sample_df, on = [:secid, :date])
D_to_save = join(D_to_save, D_clamp_df, on = [:secid, :date])
D_to_save = join(D_to_save, rn_prob_2sigma_df, on = [:secid, :date])
D_to_save = join(D_to_save, rn_prob_20mon_df, on = [:secid, :date])
D_to_save = join(D_to_save, rn_prob_40mon_df, on = [:secid, :date])

print("\n--- Outputting Data ----\n")
# CSV.write(string("estimated_data/interpolated_D/int_D_clamp_days_", days, "_", i_first, "_to_", i_last , ".csv"), D_to_save)
CSV.write(string("estimated_data/interpolated_D/int_D_clamp_days_", ARGS[1], ".csv"), D_to_save)
