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

import wrds
db = wrds.Connection()

f=open("load_single_option.sql", "r")
query = f.read()

opt_date = "'2017-09-01'"
exp_date = "'2017-10-20'"
_secid_ = '108105'

query = query.replace('\n', ' ').replace('\t', ' ')
query = query.replace('opt_date', opt_date).replace('exp_date', exp_date)
query = query.replace('_secid_', _secid_)
prot_data = db.raw_sql(query)

prot_data = prot_data.sort_values('strike_price')

option = OptionSmile(secid = prot_data['secid'].iloc[0],
					 obs_date = prot_data['date'].iloc[0],
                     exp_date = prot_data['exdate'].iloc[0],
                     spot_price = prot_data['under_price'].iloc[0],
                     strikes = np.array(prot_data['strike_price'])/1000,
                     impl_vol = np.array(prot_data['impl_volatility']))

start = time.time()
option.fit_svi_bdbg_smile()
end = time.time()
print(end-start)

ax = option.plot_svi_smile(model = 'svi_bdbg')
plt.show()

start = time.time()
option.fit_svi_var_rho_smile()
end = time.time()
print(end-start)

print(option.svi_var_rho_smile_params)

ax = option.plot_svi_smile(model = 'svi_var_rho')
plt.show()




