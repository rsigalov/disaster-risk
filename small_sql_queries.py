import numpy as np
import pandas as pd
import wrds
import os

db = wrds.Connection()
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')


query_check = """
select *
from OPTIONM.OPPRCD2017
where secid = 108105
limit 100
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)

check_data.to_csv("data/some_names.csv")




query_check = """
select *
from OPTIONM.OPTION
where optionid = 115849688
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)

query_check = """
select *
from OPTIONM.OPTION
where optionid = 115666544
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)


query_check = """
select *
from OPTIONM.ZEROCD
where date >= '2017-01-01'
and days <= 100
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)
check_data.to_csv("data/zcb_cont_comp_rate.csv")


################################################
# Loading option and distribution data for Apple
# to get a sense of how dividends work

f = open("load_option_list.sql", "r")
query = f.read()
_secid_ = '101594'
query = query.replace('\n', ' ').replace('\t', ' ')
query = query.replace('_secid_', _secid_)
opt_data = db.raw_sql(query)
opt_data.to_csv('data/opt_data_aapl.csv', index = False)


query_check = """
select 
    secid, ex_date, amount, distr_type
from OPTIONM.DISTRD
where ex_date >= '2017-01-01'
and ex_date < '2018-01-01'
and secid = 101594
"""

query_check = query_check.replace('\n', ' ').replace('\t', ' ')
check_data = db.raw_sql(query_check)

check_data.to_csv("data/dist_hist_aapl.csv")

################################################
# Loading data on dividend yield for SPX
query = """
select * 
from OPTIONM.IDXDVD 
where secid = '108105' 
and date >= '2017-01-01'
and date < '2018-01-01'
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df = db.raw_sql(query)
df.to_csv("data/spx_dividend_yield.csv")


query = """
select *
from OPTIONM.OPPRCD2017
where secid = 101594
and date = '2017-01-03'
and exdate = '2017-02-03'
and strike_price = 100000
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df = db.raw_sql(query)

############################################
# Loading option data on dates right after the Lehman collapse
# 1. Data on option prices:
f=open("load_option_list.sql", "r")
query_prices = f.read()

start_date = "'2008-09-16'"
end_date = "'2008-09-30'"
secid = '108105'
data_base = 'OPTIONM.OPPRCD2008'

query_prices = query_prices.replace('\n', ' ').replace('\t', ' ')
query_prices = query_prices.replace('_start_date_', start_date).replace('_end_date_', end_date)
query_prices = query_prices.replace('_secid_', secid)
query_prices = query_prices.replace('_data_base_', data_base)
df_prices = db.raw_sql(query_prices)

# 2. Data on index dividend yield:
query_div = """
select * 
from OPTIONM.IDXDVD 
where secid = '108105' 
and date >= '2008-09-16'
and date < '2008-09-30'
"""
query_div = query_div.replace('\n', ' ').replace('\t', ' ')
df_div = db.raw_sql(query_div)

# 3. Data on ZCB rates:
query_zcb = """
select *
from OPTIONM.ZEROCD
where date >= '2008-09-16'
and date <= '2008-09-30'
and days <= 100
"""

query_zcb = query_zcb.replace('\n', ' ').replace('\t', ' ')
df_zcb = db.raw_sql(query_zcb)

# 4. Saving data:
df_prices.to_csv('data/opt_data_lehman.csv', index = False)
df_div.to_csv('data/div_yield_lehman.csv', index = False)
df_zcb.to_csv('data/zcb_rates_lehman.csv', index = False)


f=open("load_option_list.sql", "r")
query_prices = f.read()

start_date = "'2017-01-01'"
end_date = "'2017-01-31'"
secid = "108105"
data_base = "OPTIONM.OPPRCD2017"

query_prices = query_prices.replace('\n', ' ').replace('\t', ' ')
query_prices = query_prices.replace('_start_date_', start_date).replace('_end_date_', end_date)
query_prices = query_prices.replace('_secid_', secid)
query_prices = query_prices.replace('_data_base_', data_base)
df_prices = db.raw_sql(query_prices)
df_prices.to_csv("data/opt_data_3.csv", index = False)




