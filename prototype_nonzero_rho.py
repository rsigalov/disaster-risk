########################################################
# Code for implicit minimization over (m,sigma,rho)
########################################################
import numpy as np
from numpy.linalg import inv, pinv
import pandas as pd
from pandas.tseries.offsets import *
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize, Bounds, LinearConstraint
from math import log, exp, pi, sqrt
import time
import os
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')
import wrds
db = wrds.Connection()

f=open("load_single_option.sql", "r")
query = f.read()

opt_date = "'2017-09-01'"
exp_date = "'2017-10-20'"
_secid_ = '108105'

T = (pd.to_datetime(exp_date) - pd.to_datetime(opt_date)).days/252

query = query.replace('\n', ' ').replace('\t', ' ')
query = query.replace('opt_date', opt_date).replace('exp_date', exp_date)
query = query.replace('_secid_', _secid_)
prot_data = db.raw_sql(query)

prot_data = prot_data.sort_values('strike_price')
impl_var = np.power(np.array(prot_data['impl_volatility']),2)
strike = np.array(prot_data['strike_price'])/1000

P = prot_data['under_price'].iloc[0] # Current price of the underlying
log_moneyness = np.log(strike/P)

# Supporting functions
def satisfies_constraints(beta, sigma, rho, max_v):
    a = beta[0]
    c = beta[1]
    
    satisfies = True
    if c < 0 or c > 4/(1+abs(rho)) or a < -c*sigma*sqrt(1-rho**2) or a > max_v:
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

def calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, rho, max_v, R = None, b = None):
    beta = constrained_opt(X, v, R, b)
    if satisfies_constraints(beta, sigma, rho, max_v):
        beta_opt, min_obj = compare_and_update_beta(X, v, beta, min_obj, beta_opt)
        
    return (beta_opt, min_obj)

def objective_fixed_m_sigma(m, sigma, rho, log_moneyness, impl_var, T):
    N = log_moneyness.shape[0]
    y = (log_moneyness - m)/sigma
    y_hyp = rho*sigma*y + sigma*np.sqrt(np.power(y,2) + 1)
#    v = impl_var * T
    v = impl_var
    v = v.reshape(N,-1)    
    
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
    
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, rho, max_v)
    
    ########################################################
    # 2. Looking at sides of parallelogram:
    # i. c = 0
    R = np.array([[0, 1]])
    b = np.array([[0]])
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)
    
    # ii. c = 4/(1+|rho|)
    R = np.array([[0, 1]])
    b = np.array([[4/(1+abs(rho))]])
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)

    # iii. a = -c*sigma*sqrt(1-rho^2) => a + c*sigma*sqrt(1-rho^2) = 0
    R = np.array([[1, sigma*sqrt(1-rho**2)]])
    b = np.array([[0]])
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)
    
    # iv. a = max_v
    R = np.array([[1, 0]])
    b = np.array([[max_v]])
    beta_opt, min_obj = calculate_and_update_beta(X, v, min_obj, beta_opt, sigma, rho, max_v, R, b)
    
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

def to_minimize(x):
    m = x[0]
    sigma = x[1]
    rho = x[2]
    beta_opt, min_obj = objective_fixed_m_sigma(m, sigma, rho, log_moneyness, impl_var, T)
    
    return min_obj

################################################
# Grid search to find a good starting value (need
# to experiment with grid size since it takes a lot
# of time
dim_m_grid = 15
range_m_grid = np.arange(-1,1,2/dim_m_grid)
dim_sigma_grid = 15
range_sigma_grid = np.arange(0.00001,10,(10-0.01)/dim_sigma_grid)
dim_rho_grid = 15
range_rho_grid = np.arange(-1,1,(1-(-1))/dim_rho_grid)
obj_grid = np.ones((dim_m_grid, dim_sigma_grid, dim_rho_grid))*np.Inf

start = time.time()
for i in range(dim_m_grid):
    for j in range(dim_sigma_grid):
        for k in range(dim_rho_grid):
            beta_opt, obj = objective_fixed_m_sigma(range_m_grid[i], range_sigma_grid[j],range_rho_grid[k], log_moneyness, impl_var, T)
            obj_grid[i,j,k] = obj

i_min, j_min, k_min = np.where(obj_grid == np.min(obj_grid))
m_start = range_m_grid[int(i_min)]
sigma_start = range_sigma_grid[int(j_min)]
rho_start = range_rho_grid[int(k_min)]

x0 = [m_start, sigma_start, rho_start]
#x0 = [m_start, sigma_start, 0.5]
bounds = Bounds([-np.Inf, 0.00001, -1], [np.Inf, np.Inf, 1])
opt_x = minimize(to_minimize, x0, method='SLSQP', tol=1e-12,
                 options={'ftol': 1e-12,  'maxiter': 10000},
                 bounds = bounds)
end = time.time()
print(end - start)

# Getting all the parameters for the SVI:
m_opt = opt_x['x'][0]
sigma_opt = opt_x['x'][1]
rho_opt = opt_x['x'][2]
beta_opt, obj = objective_fixed_m_sigma(m_opt, sigma_opt, rho_opt, log_moneyness, impl_var, T)
a_tilde_opt = beta_opt[0,0]
#a_opt = a_tilde_opt/T
a_opt = a_tilde_opt
c_opt = beta_opt[1,0]
#b_opt = c_opt/T
b_opt = c_opt


################################################
# Plotting fitted volatility smile
def svi_smile(log_moneyness, x):
    return x[3] + x[4]*(x[2]*(log_moneyness - x[0]) + np.sqrt(np.power(log_moneyness - x[0], 2) + x[1] ** 2))

plt.scatter(log_moneyness, impl_var, alpha = 0.5)
plt.plot(log_moneyness, svi_smile(log_moneyness, [m_opt, sigma_opt, rho_opt, a_opt, b_opt]), color = 'r')
plt.show()


def svi_smile(log_moneyness, x):
    return x[3] + x[4]*(x[2]*(log_moneyness - x[0]) + np.sqrt(np.power(log_moneyness - x[0], 2) + x[1] ** 2))

def to_minimize_v2(x):
    obj = np.sum(np.power(svi_smile(log_moneyness, x) - impl_var,2))
    return obj

start = time.time()

bounds = Bounds([-np.Inf, 0.00001, -1, -np.Inf, -np.Inf], [np.Inf, np.Inf, 1, max(impl_var), 4/T])
opt_x = minimize(to_minimize_v2, [0.02, 0.3, 0, -0.3, 1.7], method='SLSQP', tol=1e-12,
                 options={'ftol': 1e-12, 'maxiter': 10000}, bounds = bounds)
end = time.time()
print(end - start)
opt_x

plt.scatter(log_moneyness, impl_var, alpha = 0.5)
plt.plot(log_moneyness, svi_smile(log_moneyness, opt_x['x']), color = 'r')
plt.show()




