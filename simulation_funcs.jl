struct process_parameters
    theta            # long-run average volatility of S
    k                # volatility mean reversion parameters
    sigma            # volatility of volatility
    rho              # correlation between innovations to volatility and price
    r                # interest rate
    T                # maturity of the option
    spot

    lambda_small     # intensity of small jumps
    mu_small         # mean of jump size
    sigma_small      # std of jump size

    lambda_disaster  # intensity of disaster jumps
    mu_disaster      # mean of jump size
    sigma_disaster   # std of jump size
end

function simulate_paths(params, num_chunks, size_chunk, len_path)
    # unpacking all parameters:
    theta = params.theta
    k = params.k
    sigma = params.sigma
    rho = params.rho
    r = params.r
    T = params.T
    spot = params.spot

    Sigma = [1 rho;
             rho 1]

    dt = T/len_path

    lambda_small = params.lambda_small
    mu_small = params.mu_small
    sigma_small = params.sigma_small

    lambda_disaster = params.lambda_disaster
    mu_disaster = params.mu_disaster
    sigma_disaster = params.sigma_disaster

    S1 = zeros(num_chunks * size_chunk) # array to store all final values of the price
                          # on which option pricing will be based
    int_mat = zeros(num_chunks * size_chunk)

    dW_normal = Distributions.MvNormal(Sigma)
    # Simulating Poissong jumps:
    poisson_small = Poisson(lambda_small * dt)
    jump_small = Normal(mu_small, sigma_small)
    poisson_disaster = Poisson(lambda_disaster * dt)
    jump_disaster = Normal(mu_disaster, sigma_disaster)

    comp_small = lambda_small * (exp(mu_small + 0.5 * sigma_small^2) - 1)
    comp_disaster = lambda_disaster * (exp(mu_disaster + 0.5 * sigma_disaster^2) - 1)

    for i_chunk = 1:num_chunks
        # print(string("Chunk ", i_chunk, " out of ", num_chunks, "\n"))

        # Brownian component for volatility and price processes:
        dW = zeros(len_path, 2, size_chunk)
        for i_path = 1:size_chunk
            dW[:,:, i_path] = rand(dW_normal, len_path)' .* sqrt(dt)
        end

        # Poisson processes to model jumps. First simulate poisson arrival proccess
        N_small = zeros(len_path, size_chunk)
        N_disaster = zeros(len_path, size_chunk)
        for i_path = 1:size_chunk
            N_small[:, i_path] = rand(poisson_small, len_path)
            N_disaster[:, i_path] = rand(poisson_disaster, len_path)
        end

        # Now assign size of jumps to Poisson process:
        for i_path = 1:size_chunk
            num_jumps_small = length(N_small[N_small[:,i_path] .> 0, i_path])
            if num_jumps_small > 0
                N_small[N_small[:, i_path] .> 0, i_path] = rand(jump_small, num_jumps_small)
            end

            num_jumps_disaster = length(N_disaster[N_disaster[:,i_path] .> 0, i_path])
            if num_jumps_disaster > 0
                N_disaster[N_disaster[:, i_path] .> 0, i_path] = rand(jump_disaster, num_jumps_disaster)
            end
        end

        # Calculating option prices for each path:
        VS = zeros(len_path, 2, size_chunk)

        # Integral part that needs to be computed along the
        # price path
        int_mat_chunk = zeros(size_chunk)

        # 1. Initializing first values:
        VS[1,1,:] = ones(size_chunk) .* theta
        VS[1,2,:] = ones(size_chunk) .* spot

        for i_t = 2:len_path
            V_prev = VS[i_t - 1, 1, :]
            dW_V = dW[i_t, 1, :]

            V_new = V_prev .+ k .* (theta .- V_prev) .* dt .+ sigma .* sqrt.(V_prev) .* dW_V  # updating volatility
            V_new = max.(0.0, V_new)

            S_prev = VS[i_t - 1, 2, :]
            dW_S = dW[i_t, 2, :]
            S_new = S_prev .+ S_prev .* r .* dt + sqrt.(V_new) .* S_prev .* dW_S  # updating stock price

            # Doing jumps
            S_new = S_new .+ S_new .* (exp.(N_small[i_t, :]) .- 1) .- S_new .* comp_small .* dt
            S_new = S_new .+ S_new .* (exp.(N_disaster[i_t, :]) .- 1) .- S_new .* comp_disaster .* dt

            VS[i_t, 1, :] = V_new
            VS[i_t, 2, :] = S_new

            # @show size(S_new)
            # @show size(S_prev)
            # @show size(int_mat_chunk)
            int_mat_chunk .+= (S_new .- S_prev)./S_prev
        end

        # Saving the last price:
        ind_start = (i_chunk - 1) * size_chunk + 1
        ind_end = i_chunk * size_chunk
        S1[ind_start:ind_end] = VS[end, 2, :]

        # Calculating the value of the integral related to integrated variation
        # ∫dS_t/S_t-:
        int_mat[ind_start:ind_end] = int_mat_chunk

    end

    return S1, int_mat
end


function calculate_option_values(S1, params, min_strike, max_strike, num_strikes)

    strike_list = LinRange(min_strike, max_strike, num_strikes)
    put_value = zeros(num_strikes)
    call_value = zeros(num_strikes)

    # 2. Calculating mean payoff (since we are operating in a risk-neutral measure)
    for i_strike = 1:length(strike_list)
        strike = strike_list[i_strike]
        put_value[i_strike] = exp(-params.r*params.T) * mean(max.(0, strike .- S1))
        call_value[i_strike] = exp(-params.r*params.T) * mean(max.(0, S1 .- strike))
    end

    return strike_list, put_value, call_value
end

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

function calculate_implied_vol(params, strike_list, put_value, call_value)
    spot = params.spot
    r = params.r
    T = params.T
    num_strikes = length(strike_list)

    put_impl_vol = zeros(num_strikes)
    call_impl_vol = zeros(num_strikes)

    for i_strike = 1:num_strikes
        function f!(F, x)
            F[1] = BS_put_price(spot, 0, r, strike_list[i_strike], x[1], T) - put_value[i_strike]
        end

        put_impl_vol[i_strike] =  nlsolve(f!, [0.2]).zero[1]
    end

    for i_strike = 1:num_strikes
        function f!(F, x)
            F[1] = BS_call_price(spot, 0, r, strike_list[i_strike], x[1], T) - call_value[i_strike]
        end

        call_impl_vol[i_strike] = nlsolve(f!, [0.2]).zero[1]
    end

    put_impl_vol_below_spot = put_impl_vol[strike_list .<= spot]
    call_impl_vol_above_spot = call_impl_vol[strike_list .> spot]
    impl_vol = vcat(put_impl_vol_below_spot, call_impl_vol_above_spot)

    return put_impl_vol, call_impl_vol, impl_vol
end

function calc_NTM_sigma(option::OptionData)
    NTM_dist = 0.05
    NTM_index = (option.strikes .<= option.spot*(1.0 + NTM_dist)) .&
                (option.strikes .>= option.spot*(1.0 - NTM_dist))
    if sum(NTM_index) == 0
        if minimum(option.strikes) > option.spot
            sigma_NTM = option.impl_vol[1]
        elseif maximum(option.strikes) < option.spot
            sigma_NTM = option.impl_vol[end]
        else
            sigma_NTM = option.impl_vol[option.strikes .<= option.spot][end]
        end
    else
        sigma_NTM = mean(option.impl_vol[NTM_index]) * sqrt(option.T)
    end
    return sigma_NTM
end

function create_option_and_fit_svi(params, strike_list, impl_vol)
    # Getting nonmissing impl_vol:
    impl_vol_non_miss = .!(isequal.(NaN, impl_vol))
    impl_vol = impl_vol[impl_vol_non_miss]
    strike_list = strike_list[impl_vol_non_miss]

    # Creating an option object:
    option = OptionData(1234, Dates.Date("1996-02-02"), Dates.Date("1997-02-02"),
                        params.spot, strike_list, impl_vol, params.T, params.r,
                        params.spot * exp(params.r*params.T))

    svi = fit_svi_zero_rho_global(option)

    return option, svi
end

function estimate_parameters_return_Ds(option, svi)
    ests = estimate_parameters_mix(option.spot, option.int_rate, option.forward,
            option.T, calc_NTM_sigma(option), minimum(option.strikes),
            maximum(option.strikes), svi)

    D                   = ests[1] - ests[2]
    D_in_sample         = ests[3] - ests[4]
    D_bound             = ests[5] - ests[6]
    D_puts              = ests[7] - ests[8]
    D_bound_puts        = ests[9] - ests[10]
    D_deep_puts         = ests[11] - ests[12]
    D_bound_deep_puts   = ests[13] - ests[14]
    D_clamp             = ests[15] - ests[16]
    D_clamp_puts        = ests[17] - ests[18]
    D_clamp_deep_puts   = ests[19] - ests[20]

    return D, D_in_sample, D_bound, D_puts, D_bound_puts, D_deep_puts,
           D_bound_deep_puts, D_clamp, D_clamp_puts, D_clamp_deep_puts
end

function calculate_rn_prb(option, svi)
    rn_cdf_pdf_full = calc_RN_CDF_PDF(option.spot, option.int_rate, option.forward, option.T,
        svi, option.spot * 0.6, minimum(option.strikes), maximum(option.strikes), false)

    rn_cdf_pdf_clamp = calc_RN_CDF_PDF(option.spot, option.int_rate, option.forward, option.T,
        svi, option.spot * 0.6, minimum(option.strikes), maximum(option.strikes), true)

    return rn_cdf_pdf_full[1], rn_cdf_pdf_clamp[1]
end
