########################################################
# Code for implicit minimization over (m,sigma,rho)
########################################################
import numpy as np
from numpy.linalg import inv, pinv
import pandas as pd
import nlopt
from pandas.tseries.offsets import *
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize, Bounds, LinearConstraint
from math import log, exp, pi, sqrt
import time
#import os
#os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')

from OptionSmile import *

import multiprocessing

num_cores = multiprocessing.cpu_count()
print(num_cores)

opt_data = pd.read_csv('data/opt_data.csv')

opt_data['date'] = pd.to_datetime(opt_data['date'])
opt_data['exdate'] = pd.to_datetime(opt_data['exdate'])

# Getting unique pairs of observation date and expiration date:
unique_dates = opt_data.groupby(['date', 'exdate']).size()
unique_dates = unique_dates.reset_index()
unique_dates = unique_dates.drop(0, axis = 1)

option_list = []

for i_row in range(100):
    obs_date = unique_dates.iloc[i_row]['date']
    exp_date = unique_dates.iloc[i_row]['exdate']
    sub_opt_data = opt_data[(opt_data['date']==obs_date) & (opt_data['exdate']==exp_date)]
    sub_opt_data = sub_opt_data.sort_values('strike_price')
    
    i_option = OptionSmile(secid = sub_opt_data['secid'].iloc[0],
					 obs_date = sub_opt_data['date'].iloc[0],
                     exp_date = sub_opt_data['exdate'].iloc[0],
                     spot_price = sub_opt_data['under_price'].iloc[0],
                     strikes = np.array(sub_opt_data['strike_price'])/1000,
                     impl_vol = np.array(sub_opt_data['impl_volatility']))
    
    option_list.append(i_option)
    
def fit_svi_to_parallel(x):
    x.fit_svi_bdbg_smile()
    
def fit_svi_global_to_parallel(x):
    x.fit_svi_bdbg_smile_global()

print("################################################")
print("Testing Parallel Estimation")
print("")
print("1. 1 Processor, grid")
start = time.time()
pool = multiprocessing.Pool(processes = 1)
pool.map(fit_svi_to_parallel, option_list)
end = time.time()
print(end-start)
print("")
print("1. 4 Processors, grid")
start = time.time()
pool = multiprocessing.Pool(processes = 4)
pool.map(fit_svi_to_parallel, option_list)
end = time.time()
print(end-start)
print("")
print("1. 8 Processors, grid")
start = time.time()
pool = multiprocessing.Pool(processes = 8)
pool.map(fit_svi_to_parallel, option_list)
end = time.time()
print(end-start)
print("")
print("1. 1 Processor, global")
start = time.time()
pool = multiprocessing.Pool(processes = 1)
pool.map(fit_svi_global_to_parallel, option_list)
end = time.time()
print(end-start)
print("")
print("1. 4 Processors, global")
start = time.time()
pool = multiprocessing.Pool(processes = 4)
pool.map(fit_svi_global_to_parallel, option_list)
end = time.time()
print(end-start)
print("")
print("1. 8 Processor, global")
start = time.time()
pool = multiprocessing.Pool(processes = 8)
pool.map(fit_svi_global_to_parallel, option_list)
end = time.time()
print(end-start)

