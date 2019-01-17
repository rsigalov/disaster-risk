using DataFrames
using CSV
using Dates
# using PyPlot
using Distributed
@everywhere using NLopt
@everywhere include("funcs.jl")

print("Started the actual program\n")
print("\n")

df = CSV.read("data/opt_data.csv"; datarow = 2, delim = ",")
df_unique = unique(df[:, [:secid,:date,:exdate]])

max_options = 1000
option_arr = Array{OptionData, 1}(undef, max_options)

for i_option = 1:max_options
    obs_date = df_unique[:date][i_option]
    exp_date = df_unique[:exdate][i_option]
    secid = df_unique[:secid][i_option]

    df_sub = df[(df.date .== obs_date) .& (df.exdate .== exp_date) .& (df.secid .== secid), :]
    df_sub = sort!(df_sub, :strike_price)

    strikes = df_sub[:strike_price]./1000
    impl_vol = df_sub[:impl_volatility]
    spot = df_sub[:under_price][1]
    T = Dates.value(exp_date-obs_date)

    option_arr[i_option] = OptionData(secid, obs_date, exp_date, spot, strikes, impl_vol, T)
end

print("Created all options\n")

# @everywhere function fit_svi_bdbg_grid_for_pmap(option::OptionData)
#     res = fit_svi_bdbg_smile_grid(option)
#     return SVIParams(res[1], res[2], res[3], res[4], res[5], res[6], res[7])
# end
#
# @time tmp = pmap(fit_svi_bdbg_grid_for_pmap, option_arr[1:2]);
# @time svi_bdbg_grid_params_arr = pmap(fit_svi_bdbg_grid_for_pmap, option_arr);
#
# print("Done with BDBG-grid\n")
#
# @everywhere function fit_svi_bdbg_global_for_pmap(option::OptionData)
#     res = fit_svi_bdbg_smile_global(option)
#     return SVIParams(res[1], res[2], res[3], res[4], res[5], res[6], res[7])
# end
#
# @time tmp = pmap(fit_svi_bdbg_global_for_pmap, option_arr[1:2]);
# @time svi_bdbg_global_params_arr = pmap(fit_svi_bdbg_global_for_pmap, option_arr);
#
# print("Done with BDBG-global\n")
#
# @everywhere function fit_svi_var_rho_grid_for_pmap(option::OptionData)
#     res = fit_svi_var_rho_smile_grid(option)
#     return SVIParams(res[1], res[2], res[3], res[4], res[5], res[6], res[7])
# end
#
# @time tmp = pmap(fit_svi_var_rho_grid_for_pmap, option_arr[1:2]);
# @time svi_var_rho_grid_params_arr = pmap(fit_svi_var_rho_grid_for_pmap, option_arr);
#
# print("Done with VarRho-grid\n")

@everywhere function fit_svi_var_rho_global_for_pmap(option::OptionData)
    res = fit_svi_var_rho_smile_grid(option)
    return SVIParams(res[1], res[2], res[3], res[4], res[5], res[6], res[7])
end

@time tmp = pmap(fit_svi_var_rho_global_for_pmap, option_arr[1:2]);
@time svi_var_rho_global_params_arr = pmap(fit_svi_var_rho_global_for_pmap, option_arr);

print("Done with VarRho-global\n")
print("Done with everything")
