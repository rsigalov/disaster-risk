############################################################################
# File with functions for estimation of fitting SVI volatility smile
# parametrization, calculating variation, integrated variation, jump
# risk measure and other characteristics related to probability
# of a rare disaster.
#
# This particular version was written to work with Julia 0.6 that
# is supported as AMI on AWS
############################################################################

# Creating a struct to store data about option and later fit
# volatility smile
struct OptionData
    secid
	date
	exdate
	spot
	strikes
	impl_vol
    T
    int_rate # continuously compounded interest rate for
             # date of option observation
    forward # forward price calculated using dividends
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

# Function to linearly interpolate zero coupon rate to match
# the maturity of the option
function interpolate_int_rate(date_obs, date_val, zcb)
    opt_days_maturity = Dates.value(date_val - date_obs)

    # pick the last date before the obs_date. If there is such a date
    # then it will be picked:
    unique_dates = unique(zcb[:date])
    unique_dates_before = unique_dates[unique_dates .<= date_obs, :]
    date_obs = unique_dates_before[end]

    zcb_sub = zcb[zcb[:date] .== date_obs, :]
    x = zcb_sub[:days]
    y = zcb_sub[:rate]

    if opt_days_maturity < minimum(x)
        int_rate = y[1]
    else
        x1 = x[x .<= opt_days_maturity][end]
        y1 = y[x .<= opt_days_maturity][end]

        x2 = x[x .> opt_days_maturity][1]
        y2 = y[x .> opt_days_maturity][1]

        int_rate = y1 + (y2 - y1) * (opt_days_maturity - x1)/(x2-x1)
    end

    return int_rate/100
end

# Core function to fit SVI parametrization with rho = 0 (parameter that
# governs asymmetry of volatility smile, i.e. relative slopes of left and
# right wings). Fitting needs to be done with numerical optimization and
# is sensitive to starting values. Therefore, one needs to first perform
# some kind of global search to find a good starting value. In this version
# of the function we use Grid search to find a starting value for the
# local optimizer.
function fit_svi_zero_rho_grid(option::OptionData)
    log_moneyness = log.(option.strikes ./ option.spot)
    impl_var = option.impl_vol .^ 2
    T = option.T

    # Performing grid search to find good starting values for
    # numerical optimization over (m, sigma)
    dim_m_grid = 30
    range_m_grid = convert(Array, -1:((1-(-1))/(dim_m_grid-1)):1)
    dim_sigma_grid = 30
    range_sigma_grid = convert(Array,0.00001:((10-0.00001)/(dim_sigma_grid-1)):10)
    obj_grid = ones(dim_m_grid, dim_sigma_grid) .* Inf

    function to_minimize(x::Vector, grad::Vector)
        beta_opt, obj = obj_zero_rho_fix_m_sigma(x[1], x[2], log_moneyness, impl_var, T)
        return obj
    end

    for i = 1:dim_m_grid
        for j = 1:dim_sigma_grid
            obj_grid[i,j] = to_minimize([range_m_grid[i], range_sigma_grid[j]], [0, 0])
        end
    end

    index_min = findmin(obj_grid)[2]
    i_min = index_min % 30
    j_min = Int((index_min - index_min%30)/30) + 1

    m_start = range_m_grid[i_min]
    sigma_start = range_sigma_grid[j_min]
    x0 = [m_start, sigma_start]

    opt = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt, [-1, 0.0001])
    upper_bounds!(opt, [1, Inf])
    ftol_abs!(opt, 1e-12)

    min_objective!(opt, to_minimize)
    (minf,minx,ret) = optimize(opt, x0)

    m_opt = minx[1]
    sigma_opt = minx[2]
    rho_opt = 0

    # Getting optimal values of a and b implied by m and sigma:
    beta_opt, obj = obj_zero_rho_fix_m_sigma(m_opt, sigma_opt, log_moneyness, impl_var, T)
    a_opt = beta_opt[1]
    b_opt = beta_opt[2]/sigma_opt

    # Constructing SVIparams struct for outputting the result:
    return SVIParams(m_opt, sigma_opt, rho_opt, a_opt, b_opt, minf, ret)
end

# In this function we use a global maximization algorithm over a bounded support
function fit_svi_zero_rho_global(option::OptionData)
    log_moneyness = log.(option.strikes ./ option.spot)
    impl_var = option.impl_vol .^ 2
    T = option.T

    function to_minimize(x::Vector, grad::Vector)
        beta_opt, obj = obj_zero_rho_fix_m_sigma(x[1], x[2], log_moneyness, impl_var, T)
        return obj
    end

    opt1 = Opt(:GN_DIRECT_L, 2)
    lower_bounds!(opt1, [-1, 0.0001])
    upper_bounds!(opt1, [1, 10])
    ftol_abs!(opt1, 1e-12)

    min_objective!(opt1, to_minimize)
    x0 = [-0.9, 2]
    (minf,minx,ret) = optimize(opt1, x0)

    opt2 = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt2, [-1, 0.0001])
    upper_bounds!(opt2, [1, Inf])
    ftol_abs!(opt2, 1e-12)

    min_objective!(opt2, to_minimize)
    (minf,minx,ret) = optimize(opt2, minx)

    m_opt = minx[1]
    sigma_opt = minx[2]
    rho_opt = 0

    # Getting optimal values of a and b implied by m and sigma:
    beta_opt, obj = obj_zero_rho_fix_m_sigma(m_opt, sigma_opt, log_moneyness, impl_var, T)
    a_opt = beta_opt[1]
    b_opt = beta_opt[2]/sigma_opt

    return SVIParams(m_opt, sigma_opt, rho_opt, a_opt, b_opt, minf, ret)
end

# This function calculates variance for given parameters (m,sigma,rho,a,b)
# and (list of) strike k
function svi_smile(k, m, sigma, rho, a, b)
	return a .+ b.*(rho.*(k.-m) .+ sqrt.((k .- m).^2 .+ sigma.^2))
end

# This functions checks if the solution satisfies the constraints imposed
# by the SVI model
function  satisfies_constraints(sigma, beta, max_v)
    a = beta[1]
    c = beta[2]

    satisfies = true
    if c < 0 || c > 4*sigma || a < -c || a > max_v
        satisfies = false
    end

    return satisfies
end

# The following two functions estimate linear regression. The first one
# calculates unconstrained simple OLS
function constrained_opt(X, v)
    XX_inv = inv(X' * X)
    beta = XX_inv * X' * v
    return beta
end

# The second one estimates OLS when there is a linear constraint on the
# coefficients of the model
function constrained_opt(X, v, R, b)
    XX_inv = inv(X' * X)
    lambda_ = inv(R * XX_inv * R') * (b .- R * XX_inv * X' * v)
    beta = XX_inv * (X' * v + R' * lambda_)

    return beta
end

# This function calculates the least squares objective for a proposed beta (beta)
# and checks if it is lower then some other objective (min_obj). If it is
# it returns the proposed beta and the new, smaller objective
function compare_and_update_beta(X, v, beta, min_obj, beta_opt)
    obj = sum((X * beta - v).^2)
    if obj < min_obj
        beta_opt = beta
        min_obj = obj
    end

    return beta_opt, min_obj
end

# This function calculates beta, checks if the solution satisfies the constraints
# and return new value of beta if the solution both satisfies the constraints
# and gives lower objective than the previous optimal beta (beta_opt). This
# particular version deals with unconstrained OLS
function calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v)
    beta = constrained_opt(X, v)
    if satisfies_constraints(sigma, beta, max_v)
        beta_opt, min_obj = compare_and_update_beta(X, v, beta, min_obj, beta_opt)
    end

    return beta_opt, min_obj
end

# This particular version deals with constrained OLS
function calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)
    beta = constrained_opt(X, v, R, b)
    if satisfies_constraints(sigma, beta, max_v)
        beta_opt, min_obj = compare_and_update_beta(X, v, beta, min_obj, beta_opt)
    end

    return beta_opt, min_obj
end


function obj_zero_rho_fix_m_sigma(m, sigma, log_moneyness, impl_var, T)
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


########################################################
# Functions to estimate stuff
########################################################


function BS_call_price(S0, q, r, K, sigma, T)
    d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    p1 = exp(-q*T) * S0 * cdf.(Normal(), d1)
    p2 = exp(-r*T) * K * cdf.(Normal(), d2)

    return p1 - p2
end

# Calculating BS put price:
function BS_put_price(S0, q, r, K, sigma, T)
    d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    p1 = cdf.(Normal(), -d2) * K * exp(-r*T)
    p2 = cdf.(Normal(), -d1) * S0 * exp(-q*T)

    return p1 - p2
end

# Function to calculate interpolated implied volatility for a
# given OptionData and SVI interpolated volatility smile
function calc_interp_impl_vol(option::OptionData, interp_params::SVIParams, strike)
    spot = option.spot
    log_moneyness = log.(strike/spot) # SVI was interpolated as a function of
                                      # the log of the ratio of strike to
                                      # current spot price of the underlying asset

    m = interp_params.m
    sigma = interp_params.sigma
    rho = interp_params.rho
    a = interp_params.a
    b = interp_params.b

    interp_impl_var = svi_smile(log_moneyness, m, sigma, rho, a, b)

    # SVI is formulated with implie variance (sigma^2) as its value. Therefore,
    # we need to take a square root before squaring it
    return interp_impl_var .^ 0.5
end

# Function to calculate Call (Put) option value given OptionData and
# an struct with interpolation parameters:
function calc_option_value(option::OptionData, interp_params, strike, option_type)
    # Getting implied vol for this particular strike given an interpolated
    # volatility smile
    impl_vol = calc_interp_impl_vol(option, interp_params, strike)

    # Calculating Call (Put) option price
    r = option.int_rate
    F = option.forward
    T = option.T

    if option_type == "Call"
        option_price = BS_call_price.(F * exp(-r*T), 0, r,
                                      strike, impl_vol, T)
    elseif option_type == "Put"
        option_price = BS_put_price.(F * exp(-r*T), 0, r,
                                     strike, impl_vol, T)
    else
        error("option_type should be Call or Put")
    end

    return option_price
end



# function calc_V_IV_D(option::OptionData, interp_params, low_limit, high_limit)
#     spot = option.spot
#     r = option.int_rate
#     T = option.T
#
#     # 1. First define call and put option prices as functions of the strike:
#     calc_option_value_put = K -> calc_option_value(option, interp_params, K, "Put")
#     calc_option_value_call = K -> calc_option_value(option, interp_params, K, "Call")
#
#     # 2. Next define raw integrand functions. In the case that the upper limit of
#     # integration is infinite I will need to modify with change of variables
#     # that will allow me calculate an integral with inifinte limit
#
#     # These integrals contain call(strike) function
#     V1_raw = K -> 2 * (1 - log(K/spot)) * calc_option_value_call(K)/K^2
#     W1_raw = K -> (6 * log(K/spot) - 3 * (log(K/spot))^2) * calc_option_value_call(K)/K^2
#     X1_raw = K -> (12 * log(K/spot)^2 - 4 * log(K/spot)^3) * calc_option_value_call(K)/K^2
#
#     # These integrals contain put(strike) function
#     V2_raw = K -> 2 * (1 + log(spot/K)) * calc_option_value_put(K)/K^2
#     W2_raw = K -> (6 * log(spot/K) + 3 * log(spot/K)^2) * calc_option_value_put(K)/K^2
#     X2_raw = K -> (12 * log(spot/K)^2 + 4 * log(spot/K)^3) * calc_option_value_put(K)/K^2
#
#     if isequal(high_limit, Inf)
#         IV1_raw = K -> calc_option_value_call(K)/K^2
#
#         if low_limit > spot
#             # Dealing with IV. There is only one integral:
#             IV1 = t -> IV1_raw(low_limit + t/(1-t))/(1-t)^2
#
#             # Integrating to get integrated variation
#             integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
#
#             # Modifying integrands because we have an infinite upper limit of
#             # integration.
#             V1 = t -> V1_raw(low_limit + t/(1-t))/(1-t)^2
#             W1 = t -> W1_raw(low_limit + t/(1-t))/(1-t)^2
#             X1 = t -> X1_raw(low_limit + t/(1-t))/(1-t)^2
#
#             # Actually calculating these integrals
#             V = hquadrature(V1, 0, 1)[1]
#             W = hquadrature(W1, 0, 1)[1]
#             X = hquadrature(X1, 0, 1)[1]
#
#             mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24
#
#             # Computing variation:
#             variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)
#         else
#             # In this case we are dealing with both integrals with calls and puts
#             # First, dealing with integrated variation:
#             IV1 = t -> IV1_raw(spot + t/(1-t))/(1-t)^2
#
#             IV2_raw = K -> calc_option_value_put(K)/K^2
#             IV2 = t -> IV2_raw(spot * t) * spot
#
#             integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1)[1] + hquadrature(IV2, 0, 1)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
#
#             # Modifying integrands to account for infinite upper integration limit
#             V1 = t -> V1_raw(spot + t/(1-t))/(1-t)^2
#             W1 = t -> W1_raw(spot + t/(1-t))/(1-t)^2
#             X1 = t -> X1_raw(spot + t/(1-t))/(1-t)^2
#
#             V = hquadrature(V1, 0, 1)[1] + hquadrature(V2_raw, low_limit, spot)[1]
#             W = hquadrature(W1, 0, 1)[1] + hquadrature(W2_raw, low_limit, spot)[1]
#             X = hquadrature(X1, 0, 1)[1] + hquadrature(X2_raw, low_limit, spot)[1]
#
#             mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24
#
#             variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)
#
#         end
#     else
#         IV1 = K -> calc_option_value_call(K)/K^2
#         IV2 = K -> calc_option_value_put(K)/K^2
#
#         if low_limit > spot
#             integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, low_limit, high_limit)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
#
#             V = hquadrature(V1, low_limit, high_limit)[1]
#             W = hquadrature(W1, low_limit, high_limit)[1]
#             X = hquadrature(X1, low_limit, high_limit)[1]
#
#             mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24
#
#             variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)
#
#         elseif high_limit <= spot
#             integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV2, low_limit, high_limit)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
#
#             V = hquadrature(V2_raw, low_limit, high_limit)[1]
#             W = hquadrature(W2_raw, low_limit, high_limit)[1]
#             X = hquadrature(X2_raw, low_limit, high_limit)[1]
#
#             mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24
#
#             variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)
#         else
#             integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, spot, high_limit)[1] + hquadrature(IV2, low_limit, spot)[1] - exp(-r*T)*(exp(r*T)-1-r*T))
#
#             V = hquadrature(V1_raw, spot, high_limit)[1] + hquadrature(V2_raw, low_limit, spot)[1]
#             W = hquadrature(W1_raw, spot, high_limit)[1] + hquadrature(W2_raw, low_limit, spot)[1]
#             X = hquadrature(X1_raw, spot, high_limit)[1] + hquadrature(X2_raw, low_limit, spot)[1]
#
#             mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24
#
#             variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)
#
#         end
#     end
#
#     return variation, integrated_variation
# end


function calc_V_IV_D(option::OptionData, interp_params, low_limit, high_limit)
    spot = option.spot
    r = option.int_rate
    T = option.T

    # 1. First define call and put option prices as functions of the strike:
    calc_option_value_put = K -> calc_option_value(option, interp_params, K, "Put")
    calc_option_value_call = K -> calc_option_value(option, interp_params, K, "Call")

    # 2. Next define raw integrand functions. In the case that the upper limit of
    # integration is infinite I will need to modify with change of variables
    # that will allow me calculate an integral with inifinte limit

    # These integrals contain call(strike) function
    V1_raw = K -> 2 * (1 - log(K/spot)) * calc_option_value_call(K)/K^2
    W1_raw = K -> (6 * log(K/spot) - 3 * (log(K/spot))^2) * calc_option_value_call(K)/K^2
    X1_raw = K -> (12 * log(K/spot)^2 - 4 * log(K/spot)^3) * calc_option_value_call(K)/K^2

    # These integrals contain put(strike) function
    V2_raw = K -> 2 * (1 + log(spot/K)) * calc_option_value_put(K)/K^2
    W2_raw = K -> (6 * log(spot/K) + 3 * log(spot/K)^2) * calc_option_value_put(K)/K^2
    X2_raw = K -> (12 * log(spot/K)^2 + 4 * log(spot/K)^3) * calc_option_value_put(K)/K^2

    if isequal(high_limit, Inf)
        IV1_raw = K -> calc_option_value_call(K)/K^2

        if low_limit > spot
            # Dealing with IV. There is only one integral:
            IV1 = t -> IV1_raw(low_limit + t/(1-t))/(1-t)^2

            # Integrating to get integrated variation hcubature
            integrated_variation = (exp(r*T)*2/T) * (hcubature(x -> IV1(x[1]), [0], [1])[1] - exp(-r*T)*(exp(r*T)-1-r*T))

            # Modifying integrands because we have an infinite upper limit of
            # integration.
            V1 = t -> V1_raw(low_limit + t/(1-t))/(1-t)^2
            W1 = t -> W1_raw(low_limit + t/(1-t))/(1-t)^2
            X1 = t -> X1_raw(low_limit + t/(1-t))/(1-t)^2

            # Actually calculating these integrals
            V = hcubature(x -> V1(x[1]), [0], [1])[1]
            W = hcubature(x -> W1(x[1]), [0], [1])[1]
            X = hcubature(x -> X1(x[1]), [0], [1])[1]

            mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

            # Computing variation:
            variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)
        else
            # In this case we are dealing with both integrals with calls and puts
            # First, dealing with integrated variation:
            IV1 = t -> IV1_raw(spot + t/(1-t))/(1-t)^2

            IV2_raw = K -> calc_option_value_put(K)/K^2
            IV2 = t -> IV2_raw(spot * t) * spot

            integrated_variation = (exp(r*T)*2/T) * (hcubature(x -> IV1(x[1]), [0], [1])[1] + hcubature(x -> IV2(x[1]), [0], [1])[1] - exp(-r*T)*(exp(r*T)-1-r*T))

            # Modifying integrands to account for infinite upper integration limit
            V1 = t -> V1_raw(spot + t/(1-t))/(1-t)^2
            W1 = t -> W1_raw(spot + t/(1-t))/(1-t)^2
            X1 = t -> X1_raw(spot + t/(1-t))/(1-t)^2

            V = hcubature(x -> V1(x[1]), [0], [1])[1] + hcubature(x -> V2_raw(x[1]), [low_limit], [spot])[1]
            W = hcubature(x -> W1(x[1]), [0], [1])[1] + hcubature(x -> W2_raw(x[1]), [low_limit], [spot])[1]
            X = hcubature(x -> X1(x[1]), [0], [1])[1] + hcubature(x -> X2_raw(x[1]), [low_limit], [spot])[1]

            mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

            variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)

        end
    else
        IV1 = K -> calc_option_value_call(K)/K^2
        IV2 = K -> calc_option_value_put(K)/K^2

        if low_limit > spot
            integrated_variation = (exp(r*T)*2/T) * (hcubature(x -> IV1(x[1]), [low_limit], [high_limit])[1] - exp(-r*T)*(exp(r*T)-1-r*T))

            V = hcubature(x -> V1(x[1]), [low_limit], [high_limit])[1]
            W = hcubature(x -> W1(x[1]), [low_limit], [high_limit])[1]
            X = hcubature(x -> X1(x[1]), [low_limit], [high_limit])[1]

            mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

            variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)

        elseif high_limit <= spot
            integrated_variation = (exp(r*T)*2/T) * (hcubature(x -> IV2(x[1]), [low_limit], [high_limit])[1] - exp(-r*T)*(exp(r*T)-1-r*T))

            V = hcubature(x -> V2_raw(x[1]), [low_limit], [high_limit])[1]
            W = hcubature(x -> W2_raw(x[1]), [low_limit], [high_limit])[1]
            X = hcubature(x -> X2_raw(x[1]), [low_limit], [high_limit])[1]

            mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

            variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)
        else
            integrated_variation = (exp(r*T)*2/T) * (hcubature(x -> IV1(x[1]), [spot], [high_limit])[1] + hcubature(x -> IV2(x[1]), [low_limit], [spot])[1] - exp(-r*T)*(exp(r*T)-1-r*T))

            V = hcubature(x -> V1_raw(x[1]), [spot], [high_limit])[1] + hcubature(x -> V2_raw(x[1]), [low_limit], [spot])[1]
            W = hcubature(x -> W1_raw(x[1]), [spot], [high_limit])[1] + hcubature(x -> W2_raw(x[1]), [low_limit], [spot])[1]
            X = hcubature(x -> X1_raw(x[1]), [spot], [high_limit])[1] + hcubature(x -> X2_raw(x[1]), [low_limit], [spot])[1]

            mu = exp(r*T) - 1 + exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

            variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)

        end
    end

    return variation, integrated_variation
end

# Function to calculate Risk-Neutral CDF and PDF:
function calc_RN_CDF_PDF(option::OptionData, interp_params, strike)
    spot = option.spot
    r = option.int_rate
    T = option.T

    # function to calculate call option price for a specific
    # option and interpolation parameters:
    calc_specific_option_put_value = K -> calc_option_value(option, interp_params, K, "Put")

    # First derivative of put(strike) function
    der_1_put = K -> ForwardDiff.derivative(calc_specific_option_put_value, K)

    # Second derivative of call(strike) function
    der_2_put = K -> ForwardDiff.derivative(der_1_put, K)

    # Calculaing CDF and PDF:
    cdf_value = exp(r * T) * der_1_put(strike)
    pdf_value = exp(r * T) * der_2_put(strike)

    return cdf_value, pdf_value
end

function estimate_parameters(option, interp_params)
    T = option.T
    spot = option.spot

    # 0. Calculate sigma_NTM and scale it appropriately:
    NTM_dist = 0.05
    NTM_index = (option.strikes .<= option.spot*(1.0 + NTM_dist)) .&
                (option.strikes .>= option.spot*(1.0 - NTM_dist))
    if sum(NTM_index) == 0
        sigma_NTM = option.impl_vol[option.strikes .<= option.spot][end]
    else
        sigma_NTM = mean(option.impl_vol[NTM_index]) * sqrt(option.T)
    end

    # 1. Estimate V, IV using all data:
    V, IV = calc_V_IV_D(option, interp_params, 0, Inf)
    V_in_sample, IV_in_sample = calc_V_IV_D(option, interp_params, minimum(option.strikes), maximum(option.strikes))
    V_5_5, IV_5_5 = calc_V_IV_D(option, interp_params, max(0,spot*(1-5*sigma_NTM)), spot*(1+5*sigma_NTM))

    # 2. Estimate V, IV only using OTM puts. This means that we can integrate
    # from 0 to spot, from lowest strike to spot and from 1-5*sigma to spot
    V_otm, IV_otm = calc_V_IV_D(option, interp_params, 0, spot)
    V_otm_in_sample, IV_otm_in_sample = calc_V_IV_D(option, interp_params, minimum(option.strikes), spot)
    V_otm_5_5, IV_otm_5_5 = calc_V_IV_D(option, interp_params, max(0,spot*(1-5*sigma_NTM)), spot)

    # 3. Estimate V and IV using strikes for puts that are at least 1 sigma
    # away from spot. It means that we can integrate from 0 to spot*(1-sigma),
    # from lowest strike to spot*(1-sigma) and from spot*(1-5sigma) to spot*(1-sigma)
    if sigma_NTM < 1
        V_otm1, IV_otm1 = calc_V_IV_D(option, interp_params, 0, spot * (1 - sigma_NTM))
        V_otm1_in_sample, IV_otm1_in_sample = calc_V_IV_D(option, interp_params, minimum(option.strikes), spot * (1 - sigma_NTM))
        V_otm1_5_5, IV_otm1_5_5 = calc_V_IV_D(option, interp_params, max(0,spot*(1-5*sigma_NTM)), spot * (1 - sigma_NTM))
    else
        V_otm1, IV_otm1, V_otm1_in_sample, IV_otm1_in_sample, V_otm1_5_5, IV_otm1_5_5 = NaN, NaN, NaN, NaN, NaN, NaN
    end

    # 4. RN probability of two sigma drop:
    rn_prob_2sigma = calc_RN_CDF_PDF(option, interp_params, max(0, spot*(1-2*sigma_NTM)))[1]

    # 5. Need 40% annualized decline. Not sure what annualized means
    rn_prob_40ann = calc_RN_CDF_PDF(option, interp_params, max(0, spot*0.6^(1/T)))[1]

    return V, IV, V_in_sample, IV_in_sample, V_5_5, IV_5_5, V_otm, IV_otm, V_otm_in_sample, IV_otm_in_sample,
        V_otm_5_5, IV_otm_5_5, V_otm1, IV_otm1, V_otm1_in_sample, IV_otm1_in_sample, V_otm1_5_5, IV_otm1_5_5,
        rn_prob_2sigma, rn_prob_40ann
end
