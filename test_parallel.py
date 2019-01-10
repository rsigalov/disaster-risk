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

from OptionSmile import *

import multiprocessing
from joblib import Parallel, delayed

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
    

start = time.time()

def fit_svi_to_parallel(x):
    x.fit_svi_bdbg_smile()

squares = Parallel(n_jobs=4)(delayed(fit_svi_to_parallel)(x) for x in option_list[0:100])

end = time.time()
print(end-start)

