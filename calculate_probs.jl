############################################################
# Doing something with risk-neutral probabilities
############################################################

print("\n ---- Loading libraries ----\n")

using DataFrames
using Dierckx # Package for interpolation

using CSV
using Dates

days = tryparse(Float64, ARGS[1])

print("\n--- Loading Data ----\n")

# 1. reading files from the directory
dir_path = "estimated_data/V_IV/"
file_list = readdir(dir_path)
# file_list = file_list[i_first:i_last]

dir_path = "estimated_data/interpolated_D/"
file_to_load = string("int_D_clamp_days_30.csv")
df = CSV.read(string(dir_path, file_to_load); datarow = 2, delim = ",")
df[:date_adj] = Dates.lastdayofmonth.(df[:date])

# Function to trim the variable and calculate the mean
function filter_calc_mean(v, level)
    low_quant = quantile(v, level)
    upp_quant = quantile(v, 1-level)
    return mean(v[(v .>= low_quant) .& (v .<= upp_quant)])
end

############################################################
# 1. Calculating average for all firms in a month
############################################################
prob_var_list = [:rn_prob_2sigma, :rn_prob_20mon, :rn_prob_40mon]
for i = 1:length(prob_var_list)
    prob_var = prob_var_list[i]

    prob_average = by(
        df[.!isequal.(df[prob_var], NaN) .& .!isequal.(df[prob_var], Inf) .& .!isequal.(df[prob_var], -Inf),:],
        :date_adj, prob_mean = prob_var => x -> filter_calc_mean(x, 0.01))
    sort!(prob_average, :date_adj)

    if i == 1
        Plots.display(Plots.plot(prob_average.date_adj, prob_average.prob_mean, label = string(prob_var)))
    else
        Plots.display(Plots.plot!(prob_average.date_adj, prob_average.prob_mean, label = string(prob_var)))
    end
end

############################################################
# 2. Limiting to firms that are widely present in the sample
############################################################
obs_filter = 15
share_months = 0.8
prob_var_list = [:rn_prob_2sigma, :rn_prob_20ann, :rn_prob_40ann]

for i = 1:length(prob_var_list)
    @show i
    prob_var = prob_var_list[i]
    # Leaving only secid-month where there is at least 15 days of observations:
    df_filter_days = by(
        df[.!isequal.(df[prob_var], NaN) .&
           .!isequal.(df[prob_var], Inf) .&
           .!isequal.(df[prob_var], -Inf), :],
           [:secid, :date_adj],
           N = :secid => length,
           prob_mean = prob_var => x -> filter_calc_mean(x, 0.01))

    df_filter_days = df_filter_days[df_filter_days.N .>= obs_filter, :]

    # leaving companies that have >= 15 observations in at least 80% of the sample
    total_num_months = length(unique(df_filter_days[:date_adj]))
    num_months_for_secid = by(
        df_filter_days, :secid, N = :date_adj => x -> length(unique(x))
    )

    secid_list = num_months_for_secid[num_months_for_secid[:N] .>= total_num_months * share_months,:secid]

    df_filter_secid = df_filter_days[map(x -> x in secid_list, df_filter_days[:secid]), :]

    prob_average = by(
        df_filter_secid[.!isequal.(df_filter_secid[:prob_mean], NaN) .&
                        .!isequal.(df_filter_secid[:prob_mean], Inf) .&
                        .!isequal.(df_filter_secid[:prob_mean], -Inf),:],
        :date_adj, prob_mean = :prob_mean => x -> filter_calc_mean(x, 0.01))
    sort!(prob_average, :date_adj)

    if i == 1
        Plots.display(Plots.plot(prob_average.date_adj, prob_average.prob_mean, label = string(prob_var)))
    else
        Plots.display(Plots.plot!(prob_average.date_adj, prob_average.prob_mean, label = string(prob_var)))
    end
end


############################################################
# 3. Doing PCA on firms that are widely present
############################################################
