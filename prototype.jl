using DataFrames
using CSV
using Dates
using NLopt
using Plots
using PyPlot
using SparseArrays
using LinearAlgebra
# using Distributed
include("funcs.jl")

# df = CSV.read("data/opt_data.csv"; datarow = 2, delim = ",")
df = CSV.read("data/opt_data_2.csv"; datarow = 2, delim = ",")
df_unique = unique(df[:, [:secid,:date,:exdate]])

max_options = 100
option_arr = Array{OptionData, 1}(undef, max_options)

for i_option = 1:max_options
    obs_date = df_unique[:date][i_option]
    exp_date = df_unique[:exdate][i_option]
    secid = df_unique[:secid][i_option]

    df_sub = df[(df.date .== obs_date) .& (df.exdate .== exp_date) .& (df.secid .== secid), :]
    df_sub = sort!(df_sub, :strike_price)

    df_sub = unique(df_sub)

    strikes = df_sub[:strike_price]./1000
    impl_vol = df_sub[:impl_volatility]
    spot = df_sub[:under_price][1]
    T = Dates.value(exp_date-obs_date)

    option_arr[i_option] = OptionData(secid, obs_date, exp_date, spot, strikes, impl_vol, T)
end

################################################################
# Fitting SVI interpolated volatility curves
################################################################
svi_bdbg_grid_params_arr = Array{SVIParams, 1}(undef, max_options)
@time for i_option = 1:max_options
    res = fit_svi_bdbg_smile_grid(option_arr[i_option])
    svi_bdbg_grid_params_arr[i_option] = SVIParams(res[1], res[2], res[3], res[4], res[5], res[6], res[7])
end

svi_bdbg_global_params_arr = Array{SVIParams, 1}(undef, max_options)
@time for i_option = 1:max_options
    res = fit_svi_bdbg_smile_global(option_arr[i_option])
    svi_bdbg_global_params_arr[i_option] = SVIParams(res[1], res[2], res[3], res[4], res[5], res[6], res[7])
end

svi_var_rho_grid_params_arr = Array{SVIParams, 1}(undef, max_options)
@time for i_option = 1:max_options
    res = fit_svi_var_rho_smile_grid(option_arr[i_option])
    svi_var_rho_grid_params_arr[i_option] = SVIParams(res[1], res[2], res[3], res[4], res[5], res[6], res[7])
end

svi_var_rho_global_params_arr = Array{SVIParams, 1}(undef, max_options)
@time for i_option = 1:max_options
    res = fit_svi_var_rho_smile_global(option_arr[i_option])
    svi_var_rho_global_params_arr[i_option] = SVIParams(res[1], res[2], res[3], res[4], res[5], res[6], res[7])
end

################################################################
# Using PyPlot to do the right subplots
################################################################
for i_option = 1:max_options
    print(i_option)
    print("\n")
    # clf()
    # cla()
    print("fdasfa\n")
    fig = figure("An example", figsize=(10,8));
    PyPlot.axis("off")

    ax1 = fig[:add_subplot](2,2,1);
    ax2 = fig[:add_subplot](2,2,2);
    ax3 = fig[:add_subplot](2,2,3);
    ax4 = fig[:add_subplot](2,2,4);

    plot_vol_smile(option_arr[i_option], svi_bdbg_grid_params_arr[i_option], "Rho = 0, Grid", ax1);
    plot_vol_smile(option_arr[i_option], svi_bdbg_global_params_arr[i_option], "Rho = 0, Global", ax2);
    plot_vol_smile(option_arr[i_option], svi_var_rho_grid_params_arr[i_option], "Variable Rho, Grid" , ax3);
    plot_vol_smile(option_arr[i_option], svi_var_rho_global_params_arr[i_option], "Variable Rho, Global", ax4);

    subplots_adjust(wspace = 0.35, hspace = 0.35);

    title_text = string("SP500, from ", option_arr[i_option].date, " to ", option_arr[i_option].exdate);
    suptitle(title_text);
    filepath_to_save = string("images/julia_comparison/option_",i_option ,".pdf")
    PyPlot.savefig(filepath_to_save, format="pdf", bbox_inches= "tight");
end

################################################################
# Fitting volatility smile with cubic cubic splines
################################################################

cubic_splines_params_arr = map(fitCubicSpline, option_arr)

for i_option = 1:max_options
    clf()
    fig = figure("An example", figsize=(10,8));
    ax = fig[:add_subplot](1,1,1);
    plot_vol_smile(option_arr[i_option], cubic_splines_params_arr[i_option], "", ax);
    title_text = string("SP500, from ", option_arr[i_option].date, " to ", option_arr[i_option].exdate);
    suptitle(title_text);
    filepath_to_save = string("images/julia_spline_fit/option_",i_option ,".pdf")
    PyPlot.savefig(filepath_to_save, format="pdf", bbox_inches= "tight");
end



####################################################
# Testing
option = option_arr[2]

x = log.(option.strikes ./ option.spot)
n = length(x)
h = x[2:n] .- x[1:(n-1)]
sigma = option.impl_vol
sigma_diff = sigma[2:n] .- sigma[1:(n-1)]

diagA = zeros(n)
diagA[1:(n-1)] = 2*h
diagA[2:n] = diagA[2:n] + 2*h

A = spdiagm(0 => diagA, 1 => h, -1 => h)

y = zeros(n)
y[1:(n-1)] = 6 * sigma_diff ./ h
y[2:n] = y[2:n] - 6 * sigma_diff ./ h
y = hcat(y...)'

z = pinv(Matrix(A)) * y

# Calculating the actual coefficients:
d = (z[2:n] - z[1:(n-1)])./(6 * h)
c = z[1:(n-1)]./2
b = -z[1:(n-1)] .* h / 3 - z[2:n] .* h/6 + sigma_diff ./h
a = sigma[1:(n-1)]

spline = CubicSplineParams(x, a, b, c, d, sigma[1], sigma[n])


# option = option_arr[10]
#
# spline = fitCubicSpline(option)
#
# function calculateSplineVolInstance(x)
#     calculateSplineVol(x, spline)
# end
#
# clf()
# fig = figure("An example", figsize=(10,8));
# ax = fig[:add_subplot](1,1,1);
#
# log_moneyness = log.(option.strikes/option.spot)
# range_log_moneyness = log_moneyness[end] - log_moneyness[1]
#
# plot_range = LinRange(log_moneyness[1] - range_log_moneyness*0.05,
#                       log_moneyness[end] + range_log_moneyness*0.05, 1000);
#
#
# ax[:scatter](log_moneyness, option.impl_vol, alpha = 0.25, c = "b")
# ax[:plot](plot_range, map(calculateSplineVolInstance, plot_range), alpha = 0.25, c = "b")
#
# ax[:set_title]("Title")
# ax[:set_xlabel]("log(Strike/Spot)")
# ax[:set_ylabel]("Implied Variance")
#
# PyPlot.savefig("example.pdf", format = "pdf", bbox_inches = "tight");

for i_option = 1:100
    print(i_option)
    print("\n")
    fitCubicSpline(option_arr[i_option])
end

i_option =
obs_date = df_unique[:date][i_option]
exp_date = df_unique[:exdate][i_option]
secid = df_unique[:secid][i_option]

df_sub = df[(df.date .== obs_date) .& (df.exdate .== exp_date) .& (df.secid .== secid), :]
df_sub = sort!(df_sub, :strike_price)
df_sub = unique(df_sub)
