import wrds
import pandas as pd
import numpy as np
from math import sqrt, log, exp
from scipy.stats import norm
from scipy.optimize import fsolve
################################################
# Manually matching BS prices and actual OptionMetrics prices:
db = wrds.Connection()

# 1. Getting base data: best bid, best offer, implied volatility
query1 = """
select *
from OPTIONM.OPPRCD2017
where secid = 108105
and date = '2017-02-09'
and exdate = '2017-03-17'
and strike_price = 1890000
and cp_flag = 'P'
"""
query1 = query1.replace('\n', ' ').replace('\t', ' ')
df_price = db.raw_sql(query1)

# 2. Getting dividend yield for the index on observation
# date of the option
query2 = """
select *
from OPTIONM.IDXDVD 
where secid = 108105
and date = '2017-02-09'
"""
query2 = query2.replace('\n', ' ').replace('\t', ' ')
df_div_yield = db.raw_sql(query2)

# 3. Getting data on interest rate:
query3 = """
select *
from OPTIONM.ZEROCD
where date = '2017-02-09'
"""
query3 = query3.replace('\n', ' ').replace('\t', ' ')
df_zc_rate = db.raw_sql(query3)

# 4. Getting data on spot price of the index:
query4 = """
select *
from optionm.SECPRD
where date = '2017-02-09'
and secid = 108105
"""
query4 = query4.replace('\n', ' ').replace('\t', ' ')
df_spot = db.raw_sql(query4)

# 5. Putting everything together:
sigma = df_price.iloc[0]['impl_volatility'] # volatility 
strike = 1890 # strike K of the option
spot = df_spot.iloc[0]['close'] # spot price on observed date

T_days = (pd.to_datetime('2017-03-17') - pd.to_datetime('2017-02-09')).days
T = T_days/365

# Linearly interpolating zero coupon rate:
df_zc_1 = df_zc_rate[df_zc_rate['days'] <= T_days].iloc[-1]
df_zc_2 = df_zc_rate[df_zc_rate['days'] > T_days].iloc[0]

r1 = df_zc_1['rate']/100
days1 = df_zc_1['days']
r2 = df_zc_2['rate']/100
days2 = df_zc_2['days']

r = r1 + (T_days - days1) * (r2 - r1)/(days2 - days1)

q = df_div_yield.iloc[0]['rate']/100

# Calculating put option price with continuously comp. div yield
d1 = (log(spot/strike) + (r - q + 0.5*sigma**2)*T)/(sigma * sqrt(T))
d2 = d1 - sigma*sqrt(T)

put_price = exp(-r*T) * (strike * norm.cdf(-d2) - spot*exp((r-q)*T)*norm.cdf(-d1))
mid_price = 0.5*(df_price.iloc[0]['best_bid'] + df_price.iloc[0]['best_offer'])

print('')
print('Comparing actual option mid price with the one calculated')
print('using implied volatility and other parameters from OptionMetrics')
print(' * BS put option price: %.3f' % put_price)
print(' * Actual mid price:    %.3f' % mid_price)
print('')

################################################################
# Solving for implied volatility by equating put option price
# to mid price between best bid and best offer

def to_solve(sigma):
    d1 = (log(spot/strike) + (r - q + 0.5*sigma**2)*T)/(sigma * sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    put_price = exp(-r*T) * (strike * norm.cdf(-d2) - spot*exp((r-q)*T)*norm.cdf(-d1))
    
    return put_price - mid_price

sigma0 = df_price.iloc[0]['impl_volatility']
sigma_sol = fsolve(to_solve, sigma0)

print('Comparing implied vol from OptionMetrics with the one')
print('calculated on my own using parameters from OptionMetrics')
print(' * Implied vol from OptionMetrics:   %.3f' % sigma0)
print(' * Implied vol calculated on my own: %.3f' % sigma_sol[0])

    









