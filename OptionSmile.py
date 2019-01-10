import numpy as np
from numpy.linalg import inv, pinv
import pandas as pd
from pandas.tseries.offsets import *
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize, Bounds, LinearConstraint
from math import log, exp, pi, sqrt
import time

class OptionSmile:
	def __init__(self, secid, obs_date, exp_date, spot_price, strikes, impl_vol):
		self.secid = secid
		self.date = obs_date
		self.exdate = exp_date
		self.spot = spot_price
		self.strikes = strikes
		self.impl_vol = impl_vol
		self.T = (exp_date - obs_date).days/252

	def fit_svi_bdbg_smile(self):

		log_moneyness = np.log(self.strikes/self.spot)
		impl_var = np.power(self.impl_vol,2)
		T = self.T
		
		# Function for minimization over (m, sigma)
		def to_minimize(x):
		    m = x[0]
		    sigma = x[1]
		    beta_opt, min_obj = obj_bdbg_fix_m_sigma(m, sigma, log_moneyness, impl_var, T)
		    
		    return min_obj

		# Performing grid search to find good starting values for 
		# numerical optimization over (m, sigma)
		dim_m_grid = 30
		range_m_grid = np.arange(-1,1,2/dim_m_grid)
		dim_sigma_grid = 30
		range_sigma_grid = np.arange(0.00001,10,(10-0.01)/dim_sigma_grid)
		obj_grid = np.ones((dim_m_grid, dim_sigma_grid))*np.Inf

		for i in range(dim_m_grid):
		    for j in range(dim_sigma_grid):
		        beta_opt, obj = obj_bdbg_fix_m_sigma(range_m_grid[i], range_sigma_grid[j], log_moneyness, impl_var, T)
		        obj_grid[i,j] = obj

		i_min, j_min = np.where(obj_grid == np.min(obj_grid))
		m_start = range_m_grid[int(i_min)]
		sigma_start = range_sigma_grid[int(j_min)]

		# Numerical optimization over (m, sigma)
		x0 = [m_start, sigma_start]
		bounds = Bounds([-1, 0.00001], [1, np.Inf])
		opt_x = minimize(to_minimize, x0, method='SLSQP', tol=1e-12,
		                 options={'ftol': 1e-12,  'maxiter': 10000},
		                 bounds = bounds)

		# Getting all the parameters for the SVI:
		m_opt = opt_x['x'][0]
		sigma_opt = opt_x['x'][1]
		beta_opt, obj = obj_bdbg_fix_m_sigma(m_opt, sigma_opt, log_moneyness, impl_var, T)
		a_opt = beta_opt[0,0]
		c_opt = beta_opt[1,0]
		b_opt = c_opt/sigma_opt

		# Writing into the object data
		self.svi_bdbg_smile_params = {}
		self.svi_bdbg_smile_params['m'] = m_opt
		self.svi_bdbg_smile_params['sigma'] = sigma_opt
		self.svi_bdbg_smile_params['rho'] = 0
		self.svi_bdbg_smile_params['a'] = a_opt
		self.svi_bdbg_smile_params['b'] = b_opt
		self.svi_bdbg_smile_params['success'] = opt_x['success']


	def fit_svi_var_rho_smile(self):
		log_moneyness = np.log(self.strikes/self.spot)
		impl_var = np.power(self.impl_vol,2)
		T = self.T

		def to_minimize(x):
		    m = x[0]
		    sigma = x[1]
		    rho = x[2]
		    beta_opt, min_obj = obj_var_rho_fixed_m_sigma(m, sigma, rho, log_moneyness, impl_var, T)
		    
		    return min_obj

		dim_m_grid = 20
		range_m_grid = np.arange(-1,1,2/dim_m_grid)
		dim_sigma_grid = 20
		range_sigma_grid = np.arange(0.00001,10,(10-0.01)/dim_sigma_grid)
		dim_rho_grid = 20
		range_rho_grid = np.arange(-1,1,(1-(-1))/dim_rho_grid)
		obj_grid = np.ones((dim_m_grid, dim_sigma_grid, dim_rho_grid))*np.Inf

		start = time.time()
		for i in range(dim_m_grid):
		    for j in range(dim_sigma_grid):
		        for k in range(dim_rho_grid):
		            beta_opt, obj = obj_var_rho_fixed_m_sigma(range_m_grid[i], range_sigma_grid[j],range_rho_grid[k], log_moneyness, impl_var, T)
		            obj_grid[i,j,k] = obj


		i_min, j_min, k_min = np.where(obj_grid == np.min(obj_grid))
		m_start = range_m_grid[int(i_min)]
		sigma_start = range_sigma_grid[int(j_min)]
		rho_start = range_rho_grid[int(k_min)]

		x0 = [m_start, sigma_start, rho_start]
		bounds = Bounds([-1, 0.00001, -1], [1, np.Inf, 1])
		opt_x = minimize(to_minimize, x0, method='SLSQP', tol=1e-12,
		                 options={'ftol': 1e-12,  'maxiter': 10000},
		                 bounds = bounds)

		# Getting all the parameters for the SVI:
		m_opt = opt_x['x'][0]
		sigma_opt = opt_x['x'][1]
		rho_opt = opt_x['x'][2]
		beta_opt, obj = obj_var_rho_fixed_m_sigma(m_opt, sigma_opt, rho_opt, log_moneyness, impl_var, T)
		a_tilde_opt = beta_opt[0,0]
		a_opt = a_tilde_opt
		c_opt = beta_opt[1,0]
		b_opt = c_opt

		self.svi_var_rho_smile_params = {}
		self.svi_var_rho_smile_params['m'] = m_opt
		self.svi_var_rho_smile_params['sigma'] = sigma_opt
		self.svi_var_rho_smile_params['rho'] = rho_opt
		self.svi_var_rho_smile_params['a'] = a_opt
		self.svi_var_rho_smile_params['b'] = b_opt
		self.svi_var_rho_smile_params['success'] = opt_x['success']

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
def satisfies_constraints_var_rho(beta, sigma, rho, max_v):
    a = beta[0]
    c = beta[1]
    
    satisfies = True
    if c < 0 or c > 4/(1+abs(rho)) or a < -c*sigma*sqrt(1-rho**2) or a > max_v:
        satisfies = False
        
    return satisfies

def calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v, R = None, b = None):
    beta = constrained_opt(X, v, R, b)
    if satisfies_constraints_var_rho(beta, sigma, rho, max_v):
        beta_opt, min_obj = compare_and_update_beta(X, v, beta, min_obj, beta_opt)
        
    return (beta_opt, min_obj)

def obj_var_rho_fixed_m_sigma(m, sigma, rho, log_moneyness, impl_var, T):
    N = log_moneyness.shape[0]
    y = (log_moneyness - m)/sigma
    y_hyp = rho*sigma*y + sigma*np.sqrt(np.power(y,2) + 1)
    v = impl_var
    v = v.reshape(N, -1)    
    
    # Values to update if find a new maximum and it satisfies constraints 
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
    
    beta_opt, min_obj = calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v)
    
    ########################################################
    # 2. Looking at sides of parallelogram:
    # i. c = 0
    R = np.array([[0, 1]])
    b = np.array([[0]])
    beta_opt, min_obj = calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)
    
    # ii. c = 4/(1+|rho|)
    R = np.array([[0, 1]])
    b = np.array([[4/(1+abs(rho))]])
    beta_opt, min_obj = calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)

    # iii. a = -c*sigma*sqrt(1-rho^2) => a + c*sigma*sqrt(1-rho^2) = 0
    R = np.array([[1, sigma*sqrt(1-rho**2)]])
    b = np.array([[0]])
    beta_opt, min_obj = calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)
    
    # iv. a = max_v
    R = np.array([[1, 0]])
    b = np.array([[max_v]])
    beta_opt, min_obj = calculate_and_update_beta_var_rho(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)
    
    ########################################################
    # 3. Calculating objective in vertices of the constraints
    # rectangle
    # i. a = 0, c = 0
    beta_vert_1 = np.array([[0],[0]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_1, min_obj, beta_opt)
    
    # ii. a = -4*sigma*sqrt(1-rho^2)/(1+|rho|), c = 4/(1+|rho|)
    beta_vert_2 = np.array([[-4 * sigma * sqrt(1-rho**2)/(1+abs(rho))],[4/(1+abs(rho))]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_2, min_obj, beta_opt)
    
    # iii. a = max_v, c = 4sigma/(1+|rho|)
    beta_vert_3 = np.array([[max_v],[4 * sigma/(1+abs(rho))]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_3, min_obj, beta_opt)
    
    # iv. a = max_v, c = 0
    beta_vert_4 = np.array([[max_v],[0]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_4, min_obj, beta_opt)

    return (beta_opt, min_obj)

########################################################
# Supporting functions for fitting SVI with explicit
# substitution of constraints into the minimization
# objective
########################################################
# def calculate_and_update_beta_explicit_sub(X, v, min_obj, beta_opt, sigma, max_v):
#     beta = ls_opt(X, v)
#     if satisfies_constraints(sigma, beta, max_v):
#         beta_opt, min_obj = compare_and_update_beta(X, v, beta, min_obj, beta_opt)
        
#     return (beta_opt, min_obj)

# def obj_bdbg_fix_m_sigma_explicit_sub(m, sigma, log_moneyness, impl_vol, T):
#     N = log_moneyness.shape[0]
#     y = (log_moneyness - m)/sigma
#     y_hyp = np.sqrt(np.power(y,2) + 1)
#     # v = impl_vol * T
#     v = impl_vol
#     v = v.reshape(N,-1)    
    
#     # Values to store 
#     min_obj = np.Inf
#     beta_opt = np.array([[0],[0]])

#     N = v.shape[0]
#     X = np.ones((N,2))
#     X[:, 1] = y_hyp
#     max_v = max(v.flatten())
	
# 	########################################################    
#     # 1. Internal solution
#     X_fit = X
#     v_fit = v
#     beta_opt, min_obj = calculate_and_update_beta_explicit_sub(X_fit, v_fit, min_obj, beta_opt, sigma, max_v)

#     ########################################################
#     # 2. Looking at sides of parallelepipid:
#     # i. c = 0
#     X_fit = np.ones((N,1))
#     v_fit = v
#     beta_opt, min_obj = calculate_and_update_beta_explicit_sub(X_fit, v_fit, min_obj, beta_opt, sigma, max_v)
    
#     # ii. c = 4\sigma
#     v_fit = v - 4*sigma*y_hyp
#     beta_opt, min_obj = calculate_and_update_beta_explicit_sub(X_fit, v_fit, min_obj, beta_opt, sigma, max_v)

#     # iii. a = -c => a + c = 0
#     X_fit = y_hyp.reshape(N,-1) - 1
#     v_fit = v
#     beta_opt, min_obj = calculate_and_update_beta_explicit_sub(X_fit, v_fit, min_obj, beta_opt, sigma, max_v)
    
#     # iv. a = max_v
#     X_fit = y_hyp.reshape(N,-1)  
#     v_fit = v - max_v
#     beta_opt, min_obj = calculate_and_update_beta_explicit_sub(X_fit, v_fit, min_obj, beta_opt, sigma, max_v)

#     ########################################################
#     # 3. Calculating objective in vertices of the constraints
#     # rectangle
#     # i. a = 0, c = 0
#     beta_vert_1 = np.array([[0],[0]])
#     beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_1, min_obj, beta_opt)
    
#     # ii. a = -4sigma, c = 4sigma
#     beta_vert_2 = np.array([[-4 * sigma],[4 * sigma]])
#     beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_2, min_obj, beta_opt)
    
#     # iii. a = max_v, c = 0
#     beta_vert_3 = np.array([[max_v],[0]])
#     beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_3, min_obj, beta_opt)
    
#     # iv. a = max_v, c = 4sigma
#     beta_vert_4 = np.array([[max_v],[4 * sigma]])
#     beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_4, min_obj, beta_opt)

#     return (beta_opt, min_obj)











