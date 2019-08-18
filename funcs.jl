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

function fit_svi_zero_rho_global(option::OptionData)
    log_moneyness = log.(option.strikes ./ option.spot)
    impl_var = option.impl_vol .^ 2
    T = option.T

    function to_minimize(x::Vector, grad::Vector)
        beta_opt, obj = obj_bdbg_fix_m_sigma(x[1], x[2], log_moneyness, impl_var, T)
        return obj
    end

    opt1 = Opt(:GN_DIRECT_L, 2)
    lower_bounds!(opt1, [-1, 0.00001])
    upper_bounds!(opt1, [1, 10])
    ftol_abs!(opt1, 1e-12)
    maxtime!(opt1, 1) # Some optimization can get stuck there. So, we want to
                      # limit optimization time

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
    # a_opt = beta_opt[1]
    # b_opt = beta_opt[2]/sigma_opt

    a_opt = beta_opt[1]/T
    b_opt = beta_opt[2]/(sigma_opt * T)

    return SVIParams(m_opt, sigma_opt, rho_opt, a_opt, b_opt, minf, ret)
end

################################################################################
# Supporting functions for fitting SVI smile with rho = 0
################################################################################

function svi_smile(k, m, sigma, rho, a, b)
	return a .+ b.*(rho.*(k.-m) .+ sqrt.((k .- m).^2 .+ sigma.^2))
end

# Adding function to support clamping of svi-smile at endpoints
# of minimum and maximum log-moneyness
function svi_smile(k, m, sigma, rho, a, b, min_k, max_k)

    if k .< min_k
        return a .+ b.*(rho.*(min_k.-m) .+ sqrt.((min_k .- m).^2 .+ sigma.^2))
    elseif k .> max_k
        return a .+ b.*(rho.*(max_k.-m) .+ sqrt.((max_k .- m).^2 .+ sigma.^2))
    else
        return a .+ b.*(rho.*(k.-m) .+ sqrt.((k .- m).^2 .+ sigma.^2))
    end

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
    # v = impl_var
    v = impl_var * T
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
############################################################

function plot_vol_smile(option::OptionData, params::SVIParams,
                         label, ax = Missing, col_scatter = "b", col_line = "r")
    if isequal(ax, Missing)
        fig = figure("An example", figsize=(10,8));
        ax = fig[:add_subplot](1,1,1);
    end

    log_moneyness = log.(option.strikes/option.spot)
    impl_var = option.impl_vol.^2

    range_log_moneyness = log_moneyness[end] - log_moneyness[1]
    plot_range = LinRange(log_moneyness[1] - range_log_moneyness*0.20,
                          log_moneyness[end] + range_log_moneyness*0.20, 1000);

    ax[:scatter](log_moneyness, impl_var, alpha = 0.25, c = col_scatter)
    ax[:plot](plot_range, svi_smile(plot_range, params.m,
                                    params.sigma, params.rho,
                                    params.a, params.b),
              c = col_line, linewidth = 1)

    ax[:set_title](label)
    ax[:set_xlabel]("log(Strike/Spot)")
    ax[:set_ylabel]("Implied Variance")

    return ax
end

function plot_vol_smile(spot, strikes, impl_vol, params::SVIParams,
                         label, ax = Missing, col_scatter = "b", col_line = "r")
    if isequal(ax, Missing)
        fig = figure("An example", figsize=(10,8));
        ax = fig[:add_subplot](1,1,1);
    end

    log_moneyness = log.(strikes/spot)
    impl_var = impl_vol.^2

    range_log_moneyness = log_moneyness[end] - log_moneyness[1]
    plot_range = LinRange(log_moneyness[1] - range_log_moneyness*0.20,
                          log_moneyness[end] + range_log_moneyness*0.20, 1000);

    ax[:scatter](log_moneyness, impl_var, alpha = 0.25, c = col_scatter)
    ax[:plot](plot_range, svi_smile(plot_range, params.m,
                                    params.sigma, params.rho,
                                    params.a, params.b),
              c = col_line, linewidth = 1)

    ax[:set_title](label)
    ax[:set_xlabel]("log(Strike/Spot)")
    ax[:set_ylabel]("Implied Variance")

    return ax
end

########################################################################
# Functions that take not the whole OptionData struct but just parts
# of it. It seems to work considerably faster. Probably because there
# is no need to carry around all strikes and implied volatility arrays
########################################################################

################################################
# Calculating Black-Scholes Price
# function to calculate BS price for an asset with
# continuously compounded dividend at rate q. Can be
# accomodated to calculate price of option for an
# asset with discrete known ndividends
function BS_call_price(S0, q, r, K, sigma, T)
    d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    p1 = exp(-q*T) * S0 * cdf.(Normal(), d1)
    p2 = exp(-r*T) * K * cdf.(Normal(), d2)

    return p1 - p2
end

function BS_put_price(S0, q, r, K, sigma, T)
    d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    p1 = cdf.(Normal(), -d2) * K * exp(-r*T)
    p2 = cdf.(Normal(), -d1) * S0 * exp(-q*T)

    return p1 - p2
end

function calc_implied_vol(price, S0, q, r, K, T, option_type)

    if (option_type == "Call") | (option_type == "C")
        to_minimize = (x::Vector, grad::Vector) -> (BS_call_price(S0, q, r, K, x[1], T) - price)^2
    elseif (option_type == "Put") | (option_type == "P")
        to_minimize = (x::Vector, grad::Vector) -> (BS_put_price(S0, q, r, K, x[1], T) - price)^2
    else
        error("option_type should be Call or Put")
    end

    opt1 = Opt(:GN_DIRECT_L, 1)
    lower_bounds!(opt1, [0])
    upper_bounds!(opt1, [2])
    ftol_abs!(opt1, 1e-12)
    maxtime!(opt1, 1) # Some optimization can get stuck there. So, we want to
                      # limit optimization time

    min_objective!(opt1, to_minimize)
    x0 = [0.2]
    (minf,minx,ret) = optimize(opt1, x0)

    opt2 = Opt(:LN_COBYLA, 1)
    lower_bounds!(opt2, [0])
    upper_bounds!(opt2, [Inf])
    ftol_abs!(opt2, 1e-12)

    min_objective!(opt2, to_minimize)
    (minf,minx,ret) = optimize(opt2, minx)

    impl_vol_opt = minx[1]
end


# Function to calculate interpolated implied volatility for a
# given OptionData and SVI interpolated volatility smile. This first function
# interpolates vol using SVi when SVI is extrapolated (i.e. not clamped)
function calc_interp_impl_vol(spot, interp_params::SVIParams, strike)
    log_moneyness = log.(strike/spot) # SVI was interpolated as a function of
                                      # the log of the ratio of strike to
                                      # current spot price of the underlying asset

    m = interp_params.m
    sigma = interp_params.sigma
    rho = interp_params.rho
    a = interp_params.a
    b = interp_params.b

    interp_impl_var = svi_smile(log_moneyness, m, sigma, rho, a, b)

    # SVI is formulated with implied variance (sigma^2) as its value. Therefore,
    # we need to take a square root before squaring it
    return interp_impl_var .^ 0.5
end

function calc_interp_impl_vol(spot, interp_params::SVIParams, strike, min_K, max_K)
    log_moneyness = log.(strike/spot) # SVI was interpolated as a function of
                                      # the log of the ratio of strike to
                                      # current spot price of the underlying asset

    min_k = log(min_K/spot)
    max_k = log(max_K/spot)

    m = interp_params.m
    sigma = interp_params.sigma
    rho = interp_params.rho
    a = interp_params.a
    b = interp_params.b

    interp_impl_var = svi_smile(log_moneyness, m, sigma, rho, a, b, min_k, max_k)

    # SVI is formulated with implied variance (sigma^2) as its value. Therefore,
    # we need to take a square root before squaring it
    return interp_impl_var .^ 0.5
end

# Function to calculate Call (Put) option value given OptionData and
# an struct with interpolation parameters. This first function calculates
# option value when SVI is extrapolated (i.e. not clamped)
function calc_option_value(spot, r, F, T, interp_params, strike, option_type)
    # Getting implied vol for this particular strike given an interpolated
    # volatility smile
    impl_vol = calc_interp_impl_vol(spot, interp_params, strike)

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

# This second function calculates option value when SVI is clamped
function calc_option_value(spot, r, F, T, interp_params, strike, min_K, max_K, option_type)
    # Getting implied vol for this particular strike given an interpolated
    # volatility smile
    impl_vol = calc_interp_impl_vol(spot, interp_params, strike, min_K, max_K)

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

# Function to calculate Risk-Neutral CDF and PDF:
function calc_RN_CDF_PDF(spot, r, F, T, interp_params, strike, min_K, max_K, clamped = false)

    # function to calculate call option price for a specific
    # option and interpolation parameters:
    if clamped
        calc_specific_option_put_value = K -> calc_option_value(spot, r, F, T, interp_params, K, min_K, max_K, "Put")
    else
        calc_specific_option_put_value = K -> calc_option_value(spot, r, F, T, interp_params, K, "Put")
    end

    # First derivative of put(strike) function
    der_1_put = K -> ForwardDiff.derivative(calc_specific_option_put_value, K)

    # Second derivative of call(strike) function
    der_2_put = K -> ForwardDiff.derivative(der_1_put, K)

    # Calculaing CDF and PDF:
    cdf_value = exp(r * T) * der_1_put(strike)
    pdf_value = exp(r * T) * der_2_put(strike)

    return cdf_value, pdf_value
end

function calc_VIX(option::OptionData)
    r = option.int_rate
    F = option.forward
    T = option.T

    # Getting stikes that lie below (for puts) and above (fpr calls) the spot
    strikes_puts = option.strikes[option.strikes .<= option.spot]
    strikes_calls = option.strikes[option.strikes .> option.spot]

    # The same with implied volatilities
    impl_vol_puts = option.impl_vol[option.strikes .<= option.spot]
    impl_vol_calls = option.impl_vol[option.strikes .> option.spot]

    # Calculating prices for each strike and implied volatility
    calc_prices_puts = BS_put_price.(F * exp(-r*T), 0, r, strikes_puts, impl_vol_puts, T)
    calc_prices_calls = BS_call_price.(F * exp(-r*T), 0, r, strikes_calls, impl_vol_calls, T)

    strikes = option.strikes
    opt_prices = [calc_prices_puts; calc_prices_calls]
    n = length(opt_prices)
    deltaK = zeros(n)
    deltaK[1] = strikes[2]-strikes[1]
    deltaK[n] = strikes[n]-strikes[n-1]
    deltaK[2:(n-1)] = (strikes[3:n] - strikes[1:(n-2)])./2

    sigma2 = (2/T)*exp(r*T)*sum(opt_prices .* deltaK./strikes.^2) - (1/T)*(F/option.spot-1)^2
    VIX = sqrt(sigma2) * 100
    return VIX
end


function calc_V_IV_D(spot, r, F, T, interp_params, min_K, max_K, low_limit, high_limit, clamped)

    # 1. First define call and put option prices as functions of the strike. If clamped requested
    # us the corresponding function.
    if clamped
        calc_option_value_put = K -> calc_option_value(spot, r, F, T, interp_params, K, min_K, max_K, "Put")
        calc_option_value_call = K -> calc_option_value(spot, r, F, T, interp_params, K, min_K, max_K, "Call")
    else
        calc_option_value_put = K -> calc_option_value(spot, r, F, T, interp_params, K, "Put")
        calc_option_value_call = K -> calc_option_value(spot, r, F, T, interp_params, K, "Call")
    end

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
            integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1, maxevals = 100000)[1] - exp(-r*T)*(exp(r*T)-1-r*T))

            # Modifying integrands because we have an infinite upper limit of
            # integration.
            V1 = t -> V1_raw(low_limit + t/(1-t))/(1-t)^2
            W1 = t -> W1_raw(low_limit + t/(1-t))/(1-t)^2
            X1 = t -> X1_raw(low_limit + t/(1-t))/(1-t)^2

            # Actually calculating these integrals
            V = hquadrature(V1, 0, 1, maxevals = 100000)[1]
            W = hquadrature(W1, 0, 1, maxevals = 100000)[1]
            X = hquadrature(X1, 0, 1, maxevals = 100000)[1]

            mu = exp(r*T) - 1 - exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

            # Computing variation:
            variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)
        else
            # In this case we are dealing with both integrals with calls and puts
            # First, dealing with integrated variation:
            IV1 = t -> IV1_raw(spot + t/(1-t))/(1-t)^2

            IV2_raw = K -> calc_option_value_put(K)/K^2
            IV2 = t -> IV2_raw(spot * t) * spot

            integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, 0, 1, maxevals = 100000)[1] + hquadrature(IV2, 0, 1, maxevals = 100000)[1] - exp(-r*T)*(exp(r*T)-1-r*T))

            # Modifying integrands to account for infinite upper integration limit
            V1 = t -> V1_raw(spot + t/(1-t))/(1-t)^2
            W1 = t -> W1_raw(spot + t/(1-t))/(1-t)^2
            X1 = t -> X1_raw(spot + t/(1-t))/(1-t)^2

            V = hquadrature(V1, 0, 1, maxevals = 100000)[1] + hquadrature(V2_raw, low_limit, spot, maxevals = 100000)[1]
            W = hquadrature(W1, 0, 1, maxevals = 100000)[1] - hquadrature(W2_raw, low_limit, spot, maxevals = 100000)[1]
            X = hquadrature(X1, 0, 1, maxevals = 100000)[1] + hquadrature(X2_raw, low_limit, spot, maxevals = 100000)[1]

            mu = exp(r*T) - 1 - exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

            variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)

        end
    else
        IV1 = K -> calc_option_value_call(K)/K^2
        IV2 = K -> calc_option_value_put(K)/K^2

        if low_limit > spot
            integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, low_limit, high_limit, maxevals = 100000)[1] - exp(-r*T)*(exp(r*T)-1-r*T))

            V = hquadrature(V1_raw, low_limit, high_limit, maxevals = 100000)[1]
            W = hquadrature(W1_raw, low_limit, high_limit, maxevals = 100000)[1]
            X = hquadrature(X1_raw, low_limit, high_limit, maxevals = 100000)[1]

            mu = exp(r*T) - 1 - exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

            variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)

        elseif high_limit <= spot
            integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV2, low_limit, high_limit, maxevals = 100000)[1] - exp(-r*T)*(exp(r*T)-1-r*T))

            V = hquadrature(V2_raw, low_limit, high_limit, maxevals = 100000)[1]
            W = -hquadrature(W2_raw, low_limit, high_limit, maxevals = 100000)[1]
            X = hquadrature(X2_raw, low_limit, high_limit, maxevals = 100000)[1]

            mu = exp(r*T) - 1 - exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

            variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)
        else
            integrated_variation = (exp(r*T)*2/T) * (hquadrature(IV1, spot, high_limit, maxevals = 100000)[1] + hquadrature(IV2, low_limit, spot, maxevals = 100000)[1] - exp(-r*T)*(exp(r*T)-1-r*T))

            V = hquadrature(V1_raw, spot, high_limit, maxevals = 100000)[1] + hquadrature(V2_raw, low_limit, spot, maxevals = 100000)[1]
            W = hquadrature(W1_raw, spot, high_limit, maxevals = 100000)[1] - hquadrature(W2_raw, low_limit, spot, maxevals = 100000)[1]
            X = hquadrature(X1_raw, spot, high_limit, maxevals = 100000)[1] + hquadrature(X2_raw, low_limit, spot, maxevals = 100000)[1]

            mu = exp(r*T) - 1 - exp(r*T) * V/2 - exp(r*T)*W/6 - exp(r*T)*X/24

            variation = exp(r*T)*(V/T - exp(-r*T)*mu^2/T)
        end
    end

    return variation, integrated_variation
end

function estimate_parameters(spot, r, F, T, sigma_NTM, min_K, max_K, interp_params)
	
    # (1) Full SVI V and IV
    # V, IV = calc_V_IV_D(spot, r, F, T, interp_params, min_K, max_K, 0, Inf, false)
    # (2) SVI with all intergrals estimated only from the minimum to maximum
    # available strikes
    V_in_sample, IV_in_sample = calc_V_IV_D(spot, r, F, T, interp_params, min_K, max_K, min_K, max_K, false)
    # (3) SVI with clamps
    V_clamp, IV_clamp = calc_V_IV_D(spot, r, F, T, interp_params, min_K, max_K, 0, Inf, true)

    ################################################################
    # Risk-Neutral probability of a large decline
    ################################################################
    # 11. RN probability of two sigma drop:
    if sigma_NTM < 0.5
        rn_prob_2sigma = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
            max(0, spot*(1 - 2*sigma_NTM)), min_K, max_K,  false)[1]
    else
        rn_prob_2sigma = NaN
    end

    if sigma_NTM < 1
        rn_prob_sigma = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
            max(0, spot*(1 - sigma_NTM)), min_K, max_K,  false)[1]
    else
        rn_prob_sigma = NaN
    end

    # 12. Need 40% annualized decline. Not sure what annualized means
    rn_prob_20 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.2)), min_K, max_K, false)[1]
    rn_prob_40 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.4)), min_K, max_K, false)[1]
    rn_prob_60 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.6)), min_K, max_K, false)[1]
    rn_prob_80 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.8)), min_K, max_K, false)[1]

    return V_in_sample, IV_in_sample, V_clamp, IV_clamp, rn_prob_sigma,
		rn_prob_2sigma, rn_prob_20, rn_prob_40, rn_prob_60, rn_prob_80

end

# In the following function I change the magnitude of disasters specifically
# for indices since there are only a few months where S&P 500 declined by more
# than 10% and only one month where S&P declined by more than 15% -- Oct 2008
function estimate_parameters_index(spot, r, F, T, sigma_NTM, min_K, max_K, interp_params)

    # (1) Full SVI V and IV
    V, IV = calc_V_IV_D(spot, r, F, T, interp_params, min_K, max_K, 0, Inf, false)
    # (2) SVI with all intergrals estimated only from the minimum to maximum
    # available strikes
    V_in_sample, IV_in_sample = calc_V_IV_D(spot, r, F, T, interp_params, min_K, max_K, min_K, max_K, false)
    # (3) SVI with clamps
    V_clamp, IV_clamp = calc_V_IV_D(spot, r, F, T, interp_params, min_K, max_K, 0, Inf, true)

    ################################################################
    # Risk-Neutral probability of a large decline
    ################################################################
    # 11. RN probability of two sigma drop:
    if sigma_NTM < 0.5
        rn_prob_2sigma = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
            max(0, spot*(1-2*sigma_NTM)), min_K, max_K,  false)[1]
    else
        rn_prob_2sigma = NaN
    end

    if sigma_NTM < 1
        rn_prob_sigma = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
            max(0, spot*(1 - sigma_NTM)), min_K, max_K,  false)[1]
    else
        rn_prob_sigma = NaN
    end

    rn_prob_5 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.05)), min_K, max_K, false)[1]
    rn_prob_10 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.1)), min_K, max_K, false)[1]
    rn_prob_15 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.15)), min_K, max_K, false)[1]
    rn_prob_20 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.2)), min_K, max_K, false)[1]

    return V, IV, V_in_sample, IV_in_sample, V_clamp, IV_clamp, rn_prob_sigma,
        rn_prob_2sigma, rn_prob_5, rn_prob_10, rn_prob_15, rn_prob_20

end


# In the following function I change the magnitude of disasters specifically
# for indices since there are only a few months where S&P 500 declined by more
# than 10% and only one month where S&P declined by more than 15% -- Oct 2008
function estimate_parameters_rn_prob(spot, r, F, T, sigma_NTM, min_K, max_K, interp_params)

    ################################################################
    # Risk-Neutral probability of a large decline
    ################################################################
    # 11. RN probability of two sigma drop:
    if sigma_NTM < 0.5
        rn_prob_2sigma = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
            max(0, spot*(1 - 2*sigma_NTM)), min_K, max_K,  false)[1]
    else
        rn_prob_2sigma = NaN
    end

    if sigma_NTM < 1
        rn_prob_sigma = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
            max(0, spot*(1 - sigma_NTM)), min_K, max_K,  false)[1]
    else
        rn_prob_sigma = NaN
    end

    rn_prob_20 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.2)), min_K, max_K, false)[1]
    rn_prob_40 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.4)), min_K, max_K, false)[1]
    rn_prob_60 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.6)), min_K, max_K, false)[1]
    rn_prob_80 = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
        max(0, spot*(1-0.8)), min_K, max_K, false)[1]

    return rn_prob_sigma, rn_prob_2sigma, rn_prob_20, rn_prob_40,
        rn_prob_60, rn_prob_80

end

function estimate_parameters_sigma_NTM(spot, r, F, T, sigma_NTM, min_K, max_K, interp_params)

    if sigma_NTM < 0.5
        rn_prob_2sigma = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
            max(0, spot*(1 - 2*sigma_NTM)), min_K, max_K, false)[1]
    else
        rn_prob_2sigma = NaN
    end

    if sigma_NTM < 1
        rn_prob_sigma = calc_RN_CDF_PDF(spot, r, F, T, interp_params,
            max(0, spot*(1 - sigma_NTM)), min_K, max_K, false)[1]
    else
        rn_prob_sigma = NaN
    end

    return rn_prob_sigma, rn_prob_2sigma

end
