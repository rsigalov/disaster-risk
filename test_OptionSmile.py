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


query = """
select distinct
 	date, exdate
from OPTIONM.OPPRCD2017
 where secid = 108105
 and exdate - date < 40
 and exdate - date > 20
 order by date, exdate
 """
query = query.replace('\n', ' ').replace('\t', ' ')
available_option = db.raw_sql(query)
available_option.to_csv('available_sp_options', index = False)

f=open("load_single_option.sql", "r")
query = f.read()

opt_date = "'2017-06-05'"
exp_date = "'2017-07-14'"
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

# ax = option.plot_svi_smile(model = 'svi_bdbg')
# plt.show()

start = time.time()
option.fit_svi_var_rho_smile()
end = time.time()
print(end-start)

print(option.svi_var_rho_smile_params)

# ax = option.plot_svi_smile(model = 'svi_var_rho')
# plt.show()

########################################################
# Plotting side-by-side:
fig = plt.figure()
    
fig_height = 5
fig_width = 8 * 2
fig.set_figheight(fig_height)
fig.set_figwidth(fig_width)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('rho = 0')
option.plot_svi_smile(model = 'svi_bdbg', ax = ax1)

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('rho != 0')
option.plot_svi_smile(model = 'svi_var_rho', ax = ax2)

plt.show()


########################################################
# Loading many options
########################################################

f = open("load_option_list.sql", "r")
query = f.read()
_secid_ = '108105'
query = query.replace('\n', ' ').replace('\t', ' ')
query = query.replace('_secid_', _secid_)
opt_data = db.raw_sql(query)

opt_data.to_csv('data/opt_data.csv', index = False)

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

# Creating a function that will take an option and
# just call a method that fits SVI smile:
def fit_svi_to_parallel(x):
    x.fit_svi_bdbg_smile()
    
for i_row in range(100):
    option_list[i_row].fit_svi_bdbg_smile()
    print(option_list[i_row].svi_bdbg_smile_params)

    option_list[i_row].fit_svi_var_rho_smile()
    print(option_list[i_row].svi_var_rho_smile_params)
    
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 4.5)
    option_list[i_row].plot_svi_smile(model = 'svi_bdbg', ax = axs[0])
    option_list[i_row].plot_svi_smile(model = 'svi_var_rho', ax = axs[1])
    obs_date = option_list[i_row].date.strftime('%Y-%m-%d')
    exp_date = option_list[i_row].exdate.strftime('%Y-%m-%d')
    fig.suptitle('SP500, obs: %s, exp: %s' % (obs_date, exp_date), fontsize=16)
    axs[0].set_xlabel('log(Strike/Spot)')
    axs[0].set_ylabel('Implied Variance')
    axs[1].set_xlabel('log(Strike/Spot)')
    fig.savefig('images/fit_svi_bdbg_' + str(i_row) + '.pdf', bbox_inches='tight')

#    fig, ax = plt.subplots(nrows=1, ncols=1)
#    option_list[i_row].plot_svi_smile(model = 'svi_bdbg', ax = ax)
#    obs_date = option_list[i_row].date.strftime('%Y-%m-%d')
#    exp_date = option_list[i_row].exdate.strftime('%Y-%m-%d')
#    ax.set_title('SP500, obs: %s, exp: %s' % (obs_date, exp_date))
#    ax.set_xlabel('log(Strike/Spot)')
#    ax.set_ylabel('Implied Variance')
#    fig.savefig('images/fit_svi_bdbg_' + str(i_row) + '.pdf', bbox_inches='tight')

end = time.time()
print(end-start)







