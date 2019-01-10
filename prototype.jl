using DataFrames
using CSV
using Dates
using NLopt
using Plots

df = CSV.read("data/opt_data.csv"; datarow = 2, delim = ",")
df_unique = unique(df[:, [:secid,:date,:exdate]])

max_options = 100
option_arr = Array{OptionData, 1}(undef, max_options)

for i_option = 1:max_options
    obs_date = df_unique[:, :date][i_option]
    exp_date = df_unique[:, :exdate][i_option]
    secid = df_unique[:, :secid][i_option]

    df_sub = df[(df.date .== obs_date) .& (df.exdate .== exp_date) .& (df.secid .== secid), :]
    df_sub = sort!(df_sub, :strike_price)

    strikes = df_sub[:, :strike_price]./1000
    impl_vol = df_sub[:, :impl_volatility]
    spot = df_sub[:, :under_price][1]
    T = Dates.value(exp_date-obs_date)

    option_arr[i_option] = OptionData(secid, obs_date, exp_date, spot, strikes, impl_vol)
end

svi_bdbg_grid_params_arr = Array{SVIParams, 1}(undef, max_options)
for i_option = 1:max_options
    res = fit_svi_bdbg_smile_grid(option_arr[i_option])
    svi_bdbg_grid_params_arr[i_option] = SVIParams(res[1], res[2], res[3], res[4], res[5], res[6], res[7])
end

svi_bdbg_global_params_arr = Array{SVIParams, 1}(undef, max_options)
for i_option = 1:max_options
    res = fit_svi_bdbg_smile_global(option_arr[i_option])
    svi_bdbg_global_params_arr[i_option] = SVIParams(res[1], res[2], res[3], res[4], res[5], res[6], res[7])
end

for i_option = 1:max_options
    p1 = plot_svi_smile(option_arr[i_option], svi_bdbg_grid_params_arr[i_option]);
    png(p1, string("images/julia_svi_bdbg_grid/option_",i_option,".png"))

    p2 = plot_svi_smile(option_arr[i_option], svi_bdbg_global_params_arr[i_option]);
    png(p2, string("images/julia_svi_bdbg_global/option_",i_option,".png"))
end

@time fit_svi_bdbg_smile_grid(option)
@time fit_svi_bdbg_smile_global(option)

# Creating a struct to store data about option and later fit
# volatility smile
struct OptionData
    secid
	date
	exdate
	spot
	strikes
	impl_vol
end

struct SVIParams
    m
    sigma
    rho
    a
    b
    obj
    opt_result
end

function fit_svi_bdbg_smile_grid(option::OptionData)
    log_moneyness = log.(option.strikes ./ option.spot)
    impl_var = option.impl_vol .^ 2
    T = Dates.value(option.exdate - option.date)

    # Performing grid search to find good starting values for
    # numerical optimization over (m, sigma)
    dim_m_grid = 30
    range_m_grid = LinRange(-1, 1, dim_m_grid)
    dim_sigma_grid = 30
    range_sigma_grid = LinRange(0.00001, 10, dim_sigma_grid)
    obj_grid = ones(dim_m_grid, dim_sigma_grid) .* Inf

    function to_minimize(x::Vector, grad::Vector)
        beta_opt, obj = obj_bdbg_fix_m_sigma(x[1], x[2], log_moneyness, impl_var, T)
        return obj
    end

    for i = 1:dim_m_grid
        for j = 1:dim_sigma_grid
            obj_grid[i,j] = to_minimize([range_m_grid[i], range_sigma_grid[j]], [0, 0])
        end
    end

    index_min = findmin(obj_grid)[2]
    i_min = index_min[1]
    j_min = index_min[2]

    m_start = range_m_grid[i_min]
    sigma_start = range_sigma_grid[j_min]
    x0 = [m_start, sigma_start]

    opt = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt, [-1, 0.00001])
    upper_bounds!(opt, [1, Inf])
    ftol_abs!(opt, 1e-12)

    min_objective!(opt, to_minimize)
    (minf,minx,ret) = optimize(opt, x0)

    m_opt = minx[1]
    sigma_opt = minx[2]
    rho_opt = 0

    # Getting optimal values of a and b implied by m and sigma:
    beta_opt, obj = obj_bdbg_fix_m_sigma(m_opt, sigma_opt, log_moneyness, impl_var, T)
    a_opt = beta_opt[1]
    b_opt = beta_opt[2]/sigma_opt

    return m_opt, sigma_opt, rho_opt, a_opt, b_opt, minf, ret
end

function fit_svi_bdbg_smile_global(option::OptionData)
    log_moneyness = log.(option.strikes ./ option.spot)
    impl_var = option.impl_vol .^ 2
    T = Dates.value(option.exdate - option.date)

    function to_minimize(x::Vector, grad::Vector)
        beta_opt, obj = obj_bdbg_fix_m_sigma(x[1], x[2], log_moneyness, impl_var, T)
        return obj
    end

    opt1 = Opt(:GN_DIRECT_L, 2)
    lower_bounds!(opt1, [-1, 0.00001])
    upper_bounds!(opt1, [1, 10])
    ftol_abs!(opt1, 1e-12)

    min_objective!(opt1, to_minimize)
    x0 = [-0.9, 2]
    (minf,minx,ret) = optimize(opt1, x0)

    opt2 = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt2, [-1, 0.00001])
    upper_bounds!(opt2, [1, Inf])
    ftol_abs!(opt2, 1e-12)

    min_objective!(opt2, to_minimize)
    (minf,minx,ret) = optimize(opt2, minx)

    m_opt = minx[1]
    sigma_opt = minx[2]
    rho_opt = 0

    # Getting optimal values of a and b implied by m and sigma:
    beta_opt, obj = obj_bdbg_fix_m_sigma(m_opt, sigma_opt, log_moneyness, impl_var, T)
    a_opt = beta_opt[1]
    b_opt = beta_opt[2]/sigma_opt

    return m_opt, sigma_opt, rho_opt, a_opt, b_opt, minf, ret
end



################################################################################
# Supporting functions for fitting SVI smile with rho = 0
################################################################################

function svi_smile(k, m, sigma, rho, a, b)
	return a .+ b.*(rho.*(k.-m) .+ sqrt.((k .- m).^2 .+ sigma.^2))
end

function  satisfies_constraints(sigma, beta, max_v)
    a = beta[1]
    c = beta[2]

    satisfies = true
    if c < 0 || c > 4*sigma || a < -c || a > max_v
        satisfies = false
    end

    return satisfies
end

function constrained_opt(X, v, R = None, b = None)
    XX_inv = inv(X' * X)
    if isequal(R, missing) || isequal(b, missing)
        beta = XX_inv * X' * v
    else
        lambda_ = inv(R * XX_inv * R') * (b .- R * XX_inv * X' * v)
        beta = XX_inv * (X' * v + R' * lambda_)
    end

    return beta
end

function compare_and_update_beta(X, v, beta, min_obj, beta_opt)
    obj = sum((X * beta - v).^2)
    if obj < min_obj
        beta_opt = beta
        min_obj = obj
    end

    return beta_opt, min_obj
end

function calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R = missing, b = missing)
    beta = constrained_opt(X, v, R, b)
    if satisfies_constraints(sigma, beta, max_v)
        beta_opt, min_obj = compare_and_update_beta(X, v, beta, min_obj, beta_opt)
    end

    return beta_opt, min_obj
end

function obj_bdbg_fix_m_sigma(m, sigma, log_moneyness, impl_var, T)

    N = length(log_moneyness)
    y = (log_moneyness .- m)./sigma
    y_hyp = sqrt.(y.^2 .+ 1)
    v = impl_var
    v = hcat(v...)' # Transforming into 2-dim array

    min_obj = Inf
    beta_opt = zeros(2,1)

    ########################################################
    # 1. Looking for internal optimum
    # Minimizing the sum of squares (doing linear regression)
    # and checking if it satisfies no arbitrage constraints
    # on coefficients:
    X = ones(N,2)
    X[:, 2] = y_hyp
    max_v = maximum(v)

    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v)

    # If the value (and minimum of the objective) was not updated
    # it means that the solution to unconstrained LS problem doesn't
    # satisfy the constraints. Therefore, we need to continue to
    # check the sides vertices of the parameter space. If, on the
    # other hand, the objective was updated, it means that the
    # solution to unconstrained Ls problem satisfies constraints and
    # since the problem is convex it will be the global minimum.
    if isequal(min_obj, Inf)
        ########################################################
        # 2. Looking at sides of parallelepipid:
        # i. c = 0
        R = hcat([0,1]...)
        b = hcat([0]...)
        beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)

        # ii. c = 4\sigma
        R = hcat([0,1]...)
        b = hcat([4 * sigma]...)
        beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)

        # iii. a = -c => a + c = 0
        R = hcat([1, 1]...)
        b = hcat([0]...)
        beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)

        # iv. a = max_v
        R = hcat([1, 0]...)
        b = hcat([max_v]...)
        beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)

        ########################################################
        # 3. Calculating objective in vertices of the constraints
        # rectangle
        # i. a = 0, c = 0
        beta_vert_1 = hcat([0,0]...)'
        beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_1, min_obj, beta_opt)

        # ii. a = -4sigma, c = 4sigma
        beta_vert_2 = hcat([-4 * sigma,4 * sigma]...)'
        beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_2, min_obj, beta_opt)

        # iii. a = max_v, c = 0
        beta_vert_3 = hcat([max_v, 0]...)'
        beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_3, min_obj, beta_opt)

        # iv. a = max_v, c = 4sigma
        beta_vert_4 = hcat([max_v, 4 * sigma]...)'
        beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_4, min_obj, beta_opt)
    end

    return beta_opt, min_obj
end

############################################################
# Plotting the results

function plot_svi_smile(option::OptionData, svi_params::SVIParams)
    log_moneyness = log.(option.strikes/option.spot)
    impl_var = option.impl_vol.^2
    p1 = Plots.scatter(log_moneyness, impl_var, alpha = 0.5)
    Plots.plot!(log_moneyness, svi_smile(log_moneyness, svi_params.m,
                                         svi_params.sigma, svi_params.rho,
                                         svi_params.a, svi_params.b),
                title = "fdafa", lw=3, alpha = 0.7)
    xlabel!("log(Strike/Spot)")
    ylabel!("Implied Variance")
    title!(string("From ", Dates.format(option.date, "Y-m-d"), " to ", Dates.format(option.exdate, "Y-m-d")))
    return p1
end
