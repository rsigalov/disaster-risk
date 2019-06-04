print("\n ---- Loading libraries ----\n")

using DataFrames
using Dierckx # Package for interpolation
using Statistics
using Plots
using MultivariateStats
using LinearAlgebra

using CSV
using Dates

cd("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

# days = tryparse(Float64, ARGS[1])
obs_filter = 15
share_months = 0.75
print("\n--- Loading Data ----\n")


# 1. reading files from the directory
dir_path = "estimated_data/interpolated_D/"
# file_to_load = string("int_D_clamp_days_", ARGS[1])
file_to_load = string("int_D_clamp_days_30.csv")

df = CSV.read(string(dir_path, file_to_load); datarow = 2, delim = ",")

var_to_calculate = :D_clamp
# Grouping data by month and secid and filtering secid-months that have at
# least <obs_filter> observations
df[:date_adj] = Dates.lastdayofmonth.(df[:date])
num_rows = by(
    df[.!isequal.(df[var_to_calculate], NaN) .& .!isequal.(df[var_to_calculate], Inf) .& .!isequal.(df[var_to_calculate], -Inf) ,:],
    [:secid, :date_adj],
    N = :secid => length)
num_rows = num_rows[num_rows.N .>= obs_filter, :]

df = join(num_rows[:,[:secid, :date_adj]], df, on = [:secid, :date_adj], kind = :left)

function filter_calc_mean(v, level)
    low_quant = quantile(v, level)
    upp_quant = quantile(v, 1-level)
    return mean(v[(v .>= low_quant) .& (v .<= upp_quant)])
end

D_average = by(
    df[.!isequal.(df[var_to_calculate], NaN) .& .!isequal.(df[var_to_calculate], Inf) .& .!isequal.(df[var_to_calculate], -Inf),:],
    [:date_adj],
    D = [var_to_calculate] => x -> filter_calc_mean(x[var_to_calculate], 0.01))

# Averaging all firms within a month:
sort!(D_average, :date_adj)

Plots.plot(D_average.date_adj, D_average.D, label = "30 days")

# Plotting sample distribution of D within one month before and after filtering:
v = df[df.date_adj .== Dates.Date("2013-06-30"), :].D_clamp
Plots.histogram(v)
low_quant = quantile(v, 0.01)
upp_quant = quantile(v, 1-0.01)

v = v[(v .>= low_quant) .& (v .<= upp_quant)]
Plots.histogram(v)

########################################################################
# Loading data on liquidity and calculating sum of volume/open interest
# for each secid-month. Then I am going to weight by it when calculating
# average D within each month
########################################################################
# 1. Loading files:
dir_path = "estimated_data/raw_svi/"
file_list = readdir(dir_path)

file_list = file_list[occursin.("svi_params_final", file_list)]

for i in 1:length(file_list)
    @show i
    if i == 1
        global svi_df = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
        svi_df = svi_df[:, [:secid, :obs_date, :volume, :open_inetrest]]
    else
        df_append = CSV.read(string(dir_path, file_list[i]); datarow = 2, delim = ",")
        df_append = df_append[:, [:secid, :obs_date, :volume, :open_inetrest]]
        append!(svi_df, df_append)
    end
end


# 2. Calculating sum of volume and open-interest:
svi_df[:date_adj] = Dates.lastdayofmonth.(svi_df[:obs_date])

liquidity_sum = by(
    svi_df,
    [:secid,:date_adj],
    volume = :volume => sum,
    open_interest = :open_inetrest => sum)

# 3. Adding this data to within secid-month average:
D_secid_month_average = by(
    df[.!isequal.(df[var_to_calculate], NaN) .& .!isequal.(df[var_to_calculate], Inf) .& .!isequal.(df[var_to_calculate], -Inf),:],
    [:secid,:date_adj],
    D = [var_to_calculate] => x -> filter_calc_mean(x[var_to_calculate], 0.01))

D_secid_month_average = join(D_secid_month_average, liquidity_sum,
    on = [:secid, :date_adj], kind = :left)

# 4. Calculating weighted average:
function wavg(weight, v)
    return sum(weight .* v)/sum(weight)
end

weight_var = :open_interest

D_weighted_month_average = by(
    D_secid_month_average,
    [:date_adj],
    D = [weight_var, :D] => x -> wavg(x[weight_var],x[:D])
)

sort!(D_weighted_month_average, :date_adj)
Plots.plot(D_weighted_month_average.date_adj, D_weighted_month_average.D, label = "30 days")


########################################################################
# Plotting daily average to see how much noise there is
########################################################################

var_to_calculate = :D_clamp
# Grouping data by month and secid and filtering secid-months that have at
# least <obs_filter> observations
df[:date_adj] = Dates.lastdayofmonth.(df[:date])
num_rows = by(
    df[.!isequal.(df[var_to_calculate], NaN) .& .!isequal.(df[var_to_calculate], Inf) .& .!isequal.(df[var_to_calculate], -Inf) ,:],
    [:secid, :date_adj],
    N = :secid => length)
num_rows = num_rows[num_rows.N .>= obs_filter, :]

df = join(num_rows[:,[:secid, :date_adj]], df, on = [:secid, :date_adj], kind = :left)

function filter_calc_mean(v, level)
    low_quant = quantile(v, level)
    upp_quant = quantile(v, 1-level)
    return mean(v[(v .>= low_quant) .& (v .<= upp_quant)])
end

D_average = by(
    df[.!isequal.(df[var_to_calculate], NaN) .& .!isequal.(df[var_to_calculate], Inf) .& .!isequal.(df[var_to_calculate], -Inf),:],
    [:date],
    D = [var_to_calculate] => x -> filter_calc_mean(x[var_to_calculate], 0.01))

# Averaging all firms within a month:
sort!(D_average, :date)

Plots.plot(D_average.date, D_average.D, label = "30 days")


################################################################
# First, calculate average D for all companies in a given month
################################################################
dir_path = "estimated_data/interpolated_D/"
file_to_load = string("int_D_clamp_days_30.csv")
df = CSV.read(string(dir_path, file_to_load); datarow = 2, delim = ",")

df = df[:, [:secid, :date, :D_clamp]]
df[:date_adj] = Dates.lastdayofmonth.(df[:date])

var_to_calculate = :D_clamp
obs_filter = 15
share_months = 0.8

trunc_list = [0, 0.0001, 0.01]
for i = 1:length(trunc_list)
    trunc = trunc_list[i]
    # Calculating simple cross-sectional average for each month across companies:
    D_average_all = by(
        df[.!isequal.(df[var_to_calculate], NaN) .&
           .!isequal.(df[var_to_calculate], Inf) .&
           .!isequal.(df[var_to_calculate], -Inf),:],
        [:date_adj],
        D = [var_to_calculate] => x -> filter_calc_mean(x[var_to_calculate], trunc))

    sort!(D_average_all, :date_adj)

    if i == 1
        Plots.display(Plots.plot(D_average_all.date_adj,D_average_all.D, label = string(trunc), title = "Average Clamped-D"))
    else
        Plots.display(Plots.plot!(D_average_all.date_adj, D_average_all.D, label = string(trunc)))
    end
end

# Calculating median:
D_median_all = by(
    df[.!isequal.(df[var_to_calculate], NaN) .&
       .!isequal.(df[var_to_calculate], Inf) .&
       .!isequal.(df[var_to_calculate], -Inf),:],
    [:date_adj],
    med = :D_clamp => median)
sort!(D_median_all, :date_adj)
Plots.display(Plots.plot!(D_average_all.date_adj, D_median_all.med, label = "Median"))

# Calculating optimal truncation at 1% level:
D_average_all = by(
    df[.!isequal.(df[var_to_calculate], NaN) .&
       .!isequal.(df[var_to_calculate], Inf) .&
       .!isequal.(df[var_to_calculate], -Inf),:],
    [:date_adj],
    D = [var_to_calculate] => x -> filter_calc_mean(x[var_to_calculate], 0.01))
sort!(D_average_all, :date_adj)

# Calculating correlation between the two measures:
cor(D_average_all.D, D_median_all.med)


################################################################
# Next, I will filter the companies (>15days > 80% of months)
# and factor out the first principal component as well as mean
################################################################

# Leaving only secid-month's where there is at least 15 days of observations:
df_filter_days = by(
    df[.!isequal.(df[var_to_calculate], NaN) .&
       .!isequal.(df[var_to_calculate], Inf) .&
       .!isequal.(df[var_to_calculate], -Inf), :],
       [:secid, :date_adj],
       N = :secid => length,
       D = var_to_calculate => x -> filter_calc_mean(x, 0.01))

df_filter_days = df_filter_days[df_filter_days.N .>= obs_filter, :]

# leaving companies that have >= 15 observations in at least 80% of the sample
total_num_months = length(unique(df_filter_days[:date_adj]))
num_months_for_secid = by(
    df_filter_days, :secid, N = :date_adj => x -> length(unique(x))
)

secid_list = num_months_for_secid[num_months_for_secid[:N] .>= total_num_months * share_months,:secid]

df_filter_secid = df_filter_days[map(x -> x in secid_list, df_filter_days[:secid]), :]




# Calculating number of months for each company:
total_num_months = length(unique(df[:date_adj]))
num_months_for_secid = by(
    df, :secid, N = :date_adj => x -> length(unique(x))
)

secid_list = num_months_for_secid[num_months_for_secid[:N] .>= total_num_months * share_months,:secid]

df_filter_secid = df[map(x -> x in secid_list, df[:secid]), :]

df_filter_secid[:date_adj] = Dates.lastdayofmonth.(df_filter_secid[:date])
num_rows = by(
    df_filter_secid[.!isequal.(df_filter_secid[var_to_calculate], NaN) .&
                    .!isequal.(df_filter_secid[var_to_calculate], Inf) .&
                    .!isequal.(df_filter_secid[var_to_calculate], -Inf) ,:],
    [:secid, :date_adj],
    N = :secid => length)
num_rows = num_rows[num_rows.N .>= obs_filter, :]

df_filter = join(num_rows[:,[:secid, :date_adj]], df_filter_secid,
    on = [:secid, :date_adj], kind = :left)

function filter_calc_mean(v, level)
    low_quant = quantile(v, level)
    upp_quant = quantile(v, 1-level)
    return mean(v[(v .>= low_quant) .& (v .<= upp_quant)])
end

D_average = by(
    df_filter[.!isequal.(df_filter[var_to_calculate], NaN) .&
              .!isequal.(df_filter[var_to_calculate], Inf) .&
              .!isequal.(df_filter[var_to_calculate], -Inf),:],
    [:date_adj],
    D = [var_to_calculate] => x -> filter_calc_mean(x[var_to_calculate], 0.001))

# Averaging all firms within a month:
sort!(D_average, :date_adj)
Plots.plot(D_average.date_adj, D_average.D)

########################################################################
# Doing PCA on the same set of companies that are present in at least
# 80% of the months
########################################################################
# Calculating average within each secid-month:
D_secid_average = by(
    df_filter[.!isequal.(df_filter[var_to_calculate], NaN) .&
              .!isequal.(df_filter[var_to_calculate], Inf) .&
              .!isequal.(df_filter[var_to_calculate], -Inf),:],
    [:secid, :date_adj],
    D = var_to_calculate => mean)

# Pivoting the dataframe with D:
df_pivot = unstack(D_secid_average, :date_adj, :secid, :D);

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
cor_X = cor(X) # Calculating correlation matrix of D for secids
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
Plots.plot(date_vec, Dw)

# Comparing PC1 with mean (across the same SECIDs):
D_mean_filter = mean(X_D; dims = 2)

Plots.plot(date_vec, Dw, label = "PC1", title = "clamped D")
Plots.plot!(date_vec, D_mean, label = "mean D")

# Saving monthly series of PCA and mean D
df_to_output = DataFrame(
    date = date_vec[:],
    D_pc1 = Dw,
    D_mean_filter = D_mean[:],
    D_mean_all = D_mean[:])

CSV.write(string("estimated_data/disaster-risk-series/D_30.csv"), df_to_output)


########################################################################
# Doing PCA on daily data and comparing it with average to see how much
# noise there is
########################################################################

# Pivoting the dataframe with D:
df_pivot = unstack(
    df_filter[.!isequal.(df_filter[var_to_calculate], NaN) .&
              .!isequal.(df_filter[var_to_calculate], Inf) .&
              .!isequal.(df_filter[var_to_calculate], -Inf),:],
    :date, :secid, var_to_calculate);

sort!(df_pivot, :date)

# Replacing missing values with average D for each column:
N_secid = size(df_pivot)[2] - 1
for i_col = 2:(N_secid+1)
    mean_col = mean(skipmissing(df_pivot[:,i_col]))
    df_pivot[ismissing.(df_pivot[:,i_col]), i_col] = mean_col
end

# Converting dataframe to array
date_vec = df_pivot[:date]
X_D = convert(Array{Float64,2}, df_pivot[:, 2:(N_secid+1)])

# Doing PCA:
cor_X = cor(X) # Calculating correlation matrix of D for secids
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
Plots.plot(date_vec, Dw)

# Comparing PC1 with mean (across the same SECIDs):
D_mean = mean(X_D; dims = 2)

Plots.plot(date_vec, Dw, label = "PC1", title = "clamped D (daily)")
Plots.plot!(date_vec, D_mean, label = "mean D")

# Saving monthly D clamped data:
