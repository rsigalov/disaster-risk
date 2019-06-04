############################################################
# This script interpolates risk-neutral probabilities
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

for i in 1:length(file_list)
    if i == 1
        global df = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
    else
        df_append = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
        append!(df, df_append)
    end
end

# Leaving a subset of columns:
prob_var_list = [:rn_prob_2sigma, :rn_prob_20ann, :rn_prob_40ann,
                 :rn_prob_60ann, :rn_prob_80ann]
cols_to_keep = vcat([:secid, :date, :T], prob_var_list)
df = df[:,cols_to_keep]
