import numpy as np
from numpy.linalg import inv, pinv
import pandas as pd
from pandas.tseries.offsets import *
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize, Bounds, LinearConstraint
from math import log, exp, pi, sqrt
import time
import nlopt

class OptionSmile:
	def __init__(self, secid, obs_date, exp_date, spot_price, strikes, impl_vol):
		self.secid = secid
		self.date = obs_date
		self.exdate = exp_date
		self.spot = spot_price
		self.strikes = strikes
		self.impl_vol = impl_vol
		self.T = ((exp_date - obs_date).days - 1)/365

	def fit_svi(self):
		log_moneyness = np.log(self.strikes/self.spot)
		impl_var = np.power(self.impl_vol,2)
		T = self.T
		
		# Function for minimization over (m, sigma)
		def to_minimize(x, grad):
			m = x[0]
			sigma = x[1]
			beta_opt, min_obj = obj_bdbg_fix_m_sigma(m, sigma, log_moneyness, impl_var, T)

			return min_obj

		# Performing global search over a bounded parameter region
        # before doing local optimization:
		opt = nlopt.opt(nlopt.GN_DIRECT_L, 2)
		opt.set_lower_bounds([-1, 0.00001])
		opt.set_upper_bounds([1, 10])
		opt.set_min_objective(to_minimize)
		opt.set_ftol_abs(1e-12)
		opt_x = opt.optimize([-0.9, 2])

		# Numerical optimization over (m, sigma)
		x0 = [opt_x[0], opt_x[1]]

		opt = nlopt.opt(nlopt.LN_COBYLA, 2)
		opt.set_lower_bounds([-1, 0.00001])
		opt.set_upper_bounds([1, float('inf')])
		opt.set_min_objective(to_minimize)
		opt.set_ftol_abs(1e-12)
		opt_x = opt.optimize(x0)

		# Getting all the parameters for the SVI:
		m_opt = opt_x[0]
		sigma_opt = opt_x[1]
		beta_opt, obj = obj_bdbg_fix_m_sigma(m_opt, sigma_opt, log_moneyness, impl_var, T)
		a_opt = beta_opt[0,0]
		c_opt = beta_opt[1,0]
		b_opt = c_opt/sigma_opt

		# Writing into the object data
		self.svi_params = {}
		self.svi_params['m'] = m_opt
		self.svi_params['sigma'] = sigma_opt
		self.svi_params['rho'] = 0
		self.svi_params['a'] = a_opt
		self.svi_params['b'] = b_opt
		self.svi_params['success'] = opt.last_optimize_result()


	def plot_svi_smile(self, model = 'bdbg', ax = None):
		return_all = False
		if ax is None:
			return_all = True
			fig, ax = plt.subplots()

		log_moneyness = np.log(self.strikes/self.spot)
		impl_var = np.power(self.impl_vol,2)


		if model == 'svi_bdbg':
			m = self.svi_bdbg_smile_params['m']
			sigma = self.svi_bdbg_smile_params['sigma']
			rho = self.svi_bdbg_smile_params['rho']
			a = self.svi_bdbg_smile_params['a']
			b = self.svi_bdbg_smile_params['b']

			ax.scatter(log_moneyness, impl_var, alpha = 0.5)
			ax.plot(log_moneyness, svi_smile(log_moneyness, m, sigma, rho, a, b), color = 'r')

			return ax
		elif model == 'svi_var_rho':
			m = self.svi_var_rho_smile_params['m']
			sigma = self.svi_var_rho_smile_params['sigma']
			rho = self.svi_var_rho_smile_params['rho']
			a = self.svi_var_rho_smile_params['a']
			b = self.svi_var_rho_smile_params['b']

			ax.scatter(log_moneyness, impl_var, alpha = 0.5)
			ax.plot(log_moneyness, svi_smile(log_moneyness, m, sigma, rho, a, b), color = 'r')
		else:
			print('Model %s is not supported yet' % model)
            
            
    # Function to calculate interpolated implied volatility for a
    # given OptionData and SVI interpolated volatility smile
    def calc_interp_impl_vol(self, strike):
        spot = self.spot
        log_moneyness = np.log(strike/spot) # SVI was interpolated as a function of
                                          # the log of the ratio of strike to
                                          # current spot price of the underlying asset
    
        m = self.svi_params["m"]
        sigma = self.svi_params["sigma"]
        rho = self.svi_params["rho"]
        a = self.svi_params["a"]
        b = self.svi_params["b"]
    
        interp_impl_var = svi_smile(log_moneyness, m, sigma, rho, a, b)
    
        # SVI is formulated with implie variance (sigma^2) as its value. Therefore,
        # we need to take a square root before squaring it
        return np.power(interp_impl_var, 0.5)

    def calc_option_value(self, strike, option_type)
        # Getting implied vol for this particular strike given an interpolated
        # volatility smile
        impl_vol = calc_interp_impl_vol(self, strike)
    
        # Calculating Call (Put) option price
        r = self.int_rate
        F = self.forward
        T = self.T
    
        if option_type == "Call":
            option_price = BS_call_price(F * exp(-r*T), 0, r,
                                          strike, impl_vol, T)
        elif option_type == "Put":
            option_price = BS_put_price(F * exp(-r*T), 0, r,
                                         strike, impl_vol, T)
        else:
            error("option_type should be Call or Put")

    
        return option_price

    # Function to calculate Risk-Neutral CDF and PDF:
    def calc_RN_CDF(self, strike)
        spot = self.spot
        r = self.int_rate
        T = self.T
    
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
    
        return cdf_value


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



########################################################
# Supporting functions for fitting SVI with rho = 0
########################################################
def svi_smile(k, m, sigma, rho, a, b):
	return a + b*(rho*(k-m) + np.sqrt(np.power(k - m,2) + sigma**2))

def satisfies_constraints(sigma, beta, max_v):
    a = beta[0]
    c = beta[1]
    
    satisfies = True
    if c < 0 or c > 4*sigma or a < -c or a > max_v:
        satisfies = False
        
    return satisfies

def constrained_opt(X, v, R = None, b = None):
    XX_inv = pinv(X.T @ X)
    if R is None or b is None:
        beta = XX_inv @ X.T @ v
    else:
        lambda_ = pinv(R @ XX_inv @ R.T) @ (b - R @ XX_inv @ X.T @ v)
        beta = XX_inv @ (X.T @ v + R.T @ lambda_)
    
    return beta

def ls_opt(X,v):
	return pinv(X.T @ X) @ X.T @ v

def compare_and_update_beta(X, v, beta, min_obj, beta_opt):
    obj = np.sum(np.power((X @ beta).flatten() - v.flatten(),2))
    if obj < min_obj:
        beta_opt = beta
        min_obj = obj
        
    return (beta_opt, min_obj)

def calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R = None, b = None):
    beta = constrained_opt(X, v, R, b)
    if satisfies_constraints(sigma, beta, max_v):
        beta_opt, min_obj = compare_and_update_beta(X, v, beta, min_obj, beta_opt)
        
    return (beta_opt, min_obj)

def obj_bdbg_fix_m_sigma(m, sigma, log_moneyness, impl_vol, T):
    N = log_moneyness.shape[0]
    y = (log_moneyness - m)/sigma
    y_hyp = np.sqrt(np.power(y,2) + 1)
    # v = impl_vol * T
    v = impl_vol
    v = v.reshape(N,-1) 
    
    # Values to store 
    min_obj = np.Inf
    beta_opt = np.array([[0],[0]])
    
    ########################################################
    # 1. Looking for internal optimum
    # Minimizing the sum of squares (doing linear regression)
    # and checking if it satisfies no arbitrage constraints
    # on coefficients:
    N = v.shape[0]
    X = np.ones((N,2))
    X[:, 1] = y_hyp
    max_v = max(v.flatten())
    
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v)
    
    ########################################################
    # 2. Looking at sides of parallelepipid:
    # i. c = 0
    R = np.array([[0, 1]])
    b = np.array([[0]])
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)
    
    # ii. c = 4\sigma
    R = np.array([[0, 1]])
    b = np.array([[4 * sigma]])
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)

    # iii. a = -c => a + c = 0
    R = np.array([[1, 1]])
    b = np.array([[0]])
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)
    
    # iv. a = max_v
    R = np.array([[1, 0]])
    b = np.array([[max_v]])
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)
    
    ########################################################
    # 3. Calculating objective in vertices of the constraints
    # rectangle
    # i. a = 0, c = 0
    beta_vert_1 = np.array([[0],[0]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_1, min_obj, beta_opt)
    
    # ii. a = -4sigma, c = 4sigma
    beta_vert_2 = np.array([[-4 * sigma],[4 * sigma]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_2, min_obj, beta_opt)
    
    # iii. a = max_v, c = 0
    beta_vert_3 = np.array([[max_v],[0]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_3, min_obj, beta_opt)
    
    # iv. a = max_v, c = 4sigma
    beta_vert_4 = np.array([[max_v],[4 * sigma]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_4, min_obj, beta_opt)

    return (beta_opt, min_obj)

########################################################
# Supporting functions for fitting SVI with variable rho
########################################################
    
#def satisfies_constraints_var_rho(beta, sigma, rho, max_v):
#    a = beta[0]
#    c = beta[1]
#    
#    satisfies = True
#    if c < 0 or c > 4/(1+abs(rho)) or a < -c*sigma*sqrt(1-rho**2) or a > max_v:
#        satisfies = False
#        
#    return satisfies
#
#def calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v, R = None, b = None):
#    beta = constrained_opt(X, v, R, b)
#    if satisfies_constraints_var_rho(beta, sigma, rho, max_v):
#        beta_opt, min_obj = compare_and_update_beta(X, v, beta, min_obj, beta_opt)
#        
#    return (beta_opt, min_obj)
#
#def obj_var_rho_fixed_m_sigma(m, sigma, rho, log_moneyness, impl_var, T):
#    N = log_moneyness.shape[0]
#    y = (log_moneyness - m)/sigma
#    y_hyp = rho*sigma*y + sigma*np.sqrt(np.power(y,2) + 1)
#    v = impl_var
#    v = v.reshape(N, -1)    
#    
#    # Values to update if find a new maximum and it satisfies constraints 
#    min_obj = np.Inf
#    beta_opt = np.array([[0],[0]])
#    
#    ########################################################
#    # 1. Looking for internal optimum
#    # Minimizing the sum of squares (doing linear regression)
#    # and checking if it satisfies no arbitrage constraints
#    # on coefficients:
#    N = v.shape[0]
#    X = np.ones((N,2))
#    X[:, 1] = y_hyp
#    max_v = max(v.flatten())
#    
#    beta_opt, min_obj = calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v)
#    
#    ########################################################
#    # 2. Looking at sides of parallelogram:
#    # i. c = 0
#    R = np.array([[0, 1]])
#    b = np.array([[0]])
#    beta_opt, min_obj = calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)
#    
#    # ii. c = 4/(1+|rho|)
#    R = np.array([[0, 1]])
#    b = np.array([[4/(1+abs(rho))]])
#    beta_opt, min_obj = calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)
#
#    # iii. a = -c*sigma*sqrt(1-rho^2) => a + c*sigma*sqrt(1-rho^2) = 0
#    R = np.array([[1, sigma*sqrt(1-rho**2)]])
#    b = np.array([[0]])
#    beta_opt, min_obj = calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)
#    
#    # iv. a = max_v
#    R = np.array([[1, 0]])
#    b = np.array([[max_v]])
#    beta_opt, min_obj = calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)
#    
#    ########################################################
#    # 3. Calculating objective in vertices of the constraints
#    # rectangle
#    # i. a = 0, c = 0
#    beta_vert_1 = np.array([[0],[0]])
#    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_1, min_obj, beta_opt)
#    
#    # ii. a = -4*sigma*sqrt(1-rho^2)/(1+|rho|), c = 4/(1+|rho|)
#    beta_vert_2 = np.array([[-4 * sigma * sqrt(1-rho**2)/(1+abs(rho))],[4/(1+abs(rho))]])
#    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_2, min_obj, beta_opt)
#    
#    # iii. a = max_v, c = 4sigma/(1+|rho|)
#    beta_vert_3 = np.array([[max_v],[4 * sigma/(1+abs(rho))]])
#    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_3, min_obj, beta_opt)
#    
#    # iv. a = max_v, c = 0
#    beta_vert_4 = np.array([[max_v],[0]])
#    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_4, min_obj, beta_opt)
#
#    return (beta_opt, min_obj)

########################################################
# Functions for calculating option prices and integrating
########################################################

################################################
# Calculating Black-Scholes Price
# function to calculate BS price for an asset with
# continuously compounded dividend at rate q. Can be
# accomodated to calculate price of option for an
# asset with discrete known ndividends
def BS_call_price(S0, q, r, K, sigma, T):
    d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    p1 = exp(-q*T) * S0 * cdf.(Normal(), d1)
    p2 = exp(-r*T) * K * cdf.(Normal(), d2)

    return p1 - p2

def BS_put_price(S0, q, r, K, sigma, T):
    d1 = (log(S0/K) + (r - q + sigma^2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    p1 = cdf.(Normal(), -d2) * K * exp(-r*T)
    p2 = cdf.(Normal(), -d1) * S0 * exp(-q*T)

    return p1 - p2




















