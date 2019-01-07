import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
from numpy.linalg import inv, pinv
from numpy.random import normal
import random

import scipy.io as sio
from functools import reduce

import statsmodels.api as sm

from matplotlib import pyplot as plt

from datetime import datetime, timedelta

from scipy.optimize import minimize, Bounds, LinearConstraint
from math import log, exp, pi

import os
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')

import wrds
db = wrds.Connection()

query_check = """
    select distinct 
        secid, date, exdate
    from OPTIONM.OPPRCD2017
    where date = '2017-09-01'
    and secid = 101594
"""
query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)

opt_date = "'2017-09-01'"
exp_date = "'2017-10-20'"
_secid_ = '101594'

query_prot = """
    with security_table as (
        select 
            secid, date, close as under_price
        from optionm.SECPRD
        where date = opt_date
        and secid = _secid_
    )
    (
        select
    	   o.secid, o.date, o.exdate, o.cp_flag, o.strike_price, o.impl_volatility,
           (o.best_offer + o.best_bid)/2 as mid_price,
           s.under_price
    	from OPTIONM.OPPRCD2017 as o
        left join security_table as s
            on o.secid = s.secid and o.date = s.date
    	where o.secid = _secid_
        and o.cp_flag = 'C'
        and o.date = opt_date
        and o.exdate = exp_date
        and o.open_interest > 0
        and o.best_bid > 0
        and o.best_offer - o.best_bid > 0
        and o.ss_flag = '0'
        and o.delta is not null
        and o.impl_volatility is not null
        and o.strike_price/1000 > s.under_price
        and (o.best_offer + o.best_bid)/2 < s.under_price
    	order by o.exdate, o.strike_price 
    ) union (
    	select
    	   o.secid, o.date, o.exdate, o.cp_flag, o.strike_price, o.impl_volatility,
           (o.best_offer + o.best_bid)/2 as mid_price,
           s.under_price
    	from OPTIONM.OPPRCD2017 as o
        left join security_table as s
            on o.secid = s.secid and o.date = s.date
    	where o.secid = _secid_
        and o.cp_flag = 'P'
        and o.date = opt_date
        and o.exdate = exp_date
        and o.open_interest > 0
        and o.best_bid > 0
        and o.best_offer - o.best_bid > 0
        and o.ss_flag = '0'
        and o.delta is not null
        and o.impl_volatility is not null
        and o.strike_price/1000 < s.under_price
        and (o.best_offer + o.best_bid)/2 < o.strike_price/1000
        and (o.best_offer + o.best_bid)/2 >= GREATEST(0, o.strike_price/1000 - s.under_price)
    	order by o.exdate, o.strike_price
    )    
"""

query_prot = query_prot.replace('\n', ' ').replace('\t', ' ')
query_prot = query_prot.replace('opt_date', opt_date).replace('exp_date', exp_date)
query_prot = query_prot.replace('_secid_', _secid_)
prot_data = db.raw_sql(query_prot)


# Introducing list of filters for options:
strike_list = prot_data['strike_price']
impl_vol_list = prot_data['impl_volatility']

plt.scatter(strike_list, impl_vol_list)
plt.show()


############################################
# Fitting smile
a = prot_data['date'].iloc[0]
b = prot_data['exdate'].iloc[0]
delta = b - a
T = delta.days/252 # 252 trading days in a year (approximately)

# Setting sigma and m equals to 
sigma = 5
m = 0

# For given sigma and m calculate y_i as y_i = (strike_i - m)/sigma:
prot_data = prot_data.sort_values('strike_price')
impl_vol = np.array(prot_data['impl_volatility'])
strike = np.array(prot_data['strike_price'])/1000

# Calculating log-moneyness that used in SVI formulation of
# of the volatility curve:
P = prot_data['under_price'].iloc[0] # Current price of the underlying
log_moneyness = np.log(strike/P)

def satisfies_constraints(sigma, beta, max_v):
    a = beta[0]
    c = beta[1]
    
    satisfies = True
    if c < 0 or c > 4*sigma or a < 0 or a > max_v:
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

def objective_fixed_m_sigma(m, sigma, log_moneyness, impl_vol, T):
    N = log_moneyness.shape[0]
    y = (log_moneyness - m)/sigma
    y_hyp = np.sqrt(np.power(y,2) + 1)
    v = impl_vol * T
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
    
    # iii. a = 0
    R = np.array([[1, 0]])
    b = np.array([[0]])
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)
    
    # iv. a = max_v
    R = np.array([[1, 0]])
    b = np.array([[max_v]])
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, max_v, R, b)
    
    ########################################################
    # 3. Calculating objective in vertices of the constraints
    # rectangle
    beta_vert_1 = np.array([[0],[0]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_1, min_obj, beta_opt)
    
    beta_vert_2 = np.array([[4*sigma],[0]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_2, min_obj, beta_opt)
    
    beta_vert_3 = np.array([[0],[max_v]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_3, min_obj, beta_opt)
    
    beta_vert_4 = np.array([[4*sigma],[max_v]])
    beta_opt, min_obj = compare_and_update_beta(X, v, beta_vert_4, min_obj, beta_opt)

    return (beta_opt, min_obj)

def to_minimize(x):
    m = x[0]
    sigma = x[1]
    beta_opt, min_obj = objective_fixed_m_sigma(m, sigma, log_moneyness, impl_vol, T)
    
    return min_obj

################################################
# Grid search to find a good starting value:
dim_m_grid = 10
range_m_grid = np.arange(-1,1,2/dim_m_grid)
dim_sigma_grid = 10
range_sigma_grid = np.arange(0.001,10,(10-0.01)/dim_sigma_grid)
obj_grid = np.ones((dim_m_grid, dim_sigma_grid))*np.Inf

for i in range(dim_m_grid):
    for j in range(dim_sigma_grid):
        beta_opt, obj = objective_fixed_m_sigma(range_m_grid[i], range_sigma_grid[j], log_moneyness, impl_vol, T)
        obj_grid[i,j] = obj

i_min, j_min = np.where(obj_grid == np.min(obj_grid))
m_start = range_m_grid[int(i_min)]
sigma_start = range_sigma_grid[int(j_min)]

x0 = [m_start, sigma_start]
bounds = Bounds([-np.Inf, 0.00001], [np.Inf, np.Inf])
opt_x = minimize(to_minimize, x0, method='L-BFGS-B', tol=1e-12,
                 options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 10000},
                 bounds = bounds)

# Getting all the parameters for the SVI:
m_opt = opt_x['x'][0]
sigma_opt = opt_x['x'][0]
beta_opt, obj = objective_fixed_m_sigma(m_opt, sigma_opt, log_moneyness, impl_vol, T)
a_tilde_opt = beta_opt[0,0]
a_opt = a_tilde_opt/T
c_opt = beta_opt[1,0]
b_opt = c_opt/(sigma_opt * T)

################################################
# Plotting fitted volatility smile
def svi_smile(log_moneyness, x):
    return x[2] + x[3]*np.sqrt(np.power(log_moneyness - x[0], 2) + x[1] ** 2)

plt.scatter(strike, impl_vol, alpha = 0.5)
plt.plot(strike, svi_smile(log_moneyness, [m_opt, sigma_opt, a_opt, b_opt]), color = 'r')
plt.show()










################################################
# Explicit minimization over 4 variables:

def svi_smile(log_moneyness, x):
    return x[2] + x[3]*np.sqrt(np.power(log_moneyness - x[0], 2) + x[1] ** 2)

def to_minimize_v2(x):
    obj = np.sum(np.power(svi_smile(log_moneyness, x) - impl_vol,2))
    return obj

bounds = Bounds([-np.Inf, 0.00001, 0, -np.Inf], [np.Inf, np.Inf, max(impl_vol), 4/T])
opt_x = minimize(to_minimize_v2, [0.02, 0.3, -0.3, 1.7], method='SLSQP', tol=1e-12,
                 options={'ftol': 1e-12, 'maxiter': 10000}, bounds = bounds)


################################################
# Explicit minimization for fixed m and sigma
# and modified variables:
N = log_moneyness.shape[0]
y = (log_moneyness - m_opt)/sigma_opt
y_hyp = np.sqrt(np.power(y,2) + 1)
v = impl_vol * T

def to_minimize_v3(x):    
    obj = np.sum(np.power(x[0] + x[1] * y_hyp - v, 2))
    return obj

bounds = Bounds([0, 0], [max(v.flatten()), 4*sigma_opt])
opt_x = minimize(to_minimize_v3, [0.008, 0.03/70], method='SLSQP', tol=1e-12,
                 options={'ftol': 1e-12, 'maxiter': 10000}, bounds = bounds)
opt_x

plt.scatter(log_moneyness, impl_vol)
plt.plot(log_moneyness, svi_smile(log_moneyness, opt_x['x']), color = 'r')
plt.show()

plt.scatter(log_moneyness, impl_vol)
plt.plot(log_moneyness, svi_smile(log_moneyness, [m_opt, sigma_opt, 0.01470286/T, 0]), color = 'r')
plt.show()

########################################
beta = opt_x['x'].reshape((2,-1))

compare_and_update_beta(X, v, beta, np.Inf, np.array([[0],[0]]))






