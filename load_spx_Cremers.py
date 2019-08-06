"""
This scripts loads raw option data for SPX from OptionMetrics, dividend yield
and interest rates and prepares it to construct a factor from Cremers et al.
"""
import numpy as np
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance
from statsmodels.iolib.summary2 import summary_col # For summarizing regression results
import os
from matplotlib import pyplot as plt
from functools import reduce
from scipy.stats import norm
from scipy.optimize import minimize
import time

os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

import wrds
db = wrds.Connection(wrds_username = "rsigalov")

pd.set_option('display.max_columns', None)



secid = 108105
year_start = 1996
year_end = 2017

print("")
print("--- Start loading option data ---")
print("")
start = time.time()

df_prices = pd.DataFrame(columns = ["secid", "date", "exdate", "cp_flag", "strike_price",
                                    "impl_volatility", "best_offer", "best_bid", "price",
                                    "spot"])

year_list = list(range(year_start, year_end + 1, 1))
num_years = len(year_list)
i_year = 0
for year in year_list:
    i_year += 1
    print("Year %s, %s/%s" % ( str(year),str(i_year), str(num_years)))

    start_date = "'" + str(year) + "-01-01'"
    end_date = "'" + str(year) + "-12-31'"
    data_base = "OPTIONM.OPPRCD" + str(year)

    f = open("load_options_spx_cremers.sql", "r")
    query = f.read()

    query = query.replace('\n', ' ').replace('\t', ' ')
    query = query.replace('_start_date_', start_date).replace('_end_date_', end_date)
    query = query.replace('_secid_', str(secid))
    query = query.replace('_data_base_', data_base)

    df_option_i = db.raw_sql(query)
    df_prices = df_prices.append(df_option_i)

end = time.time()
print("")
print("--- Time to load option data ---")
print(end - start)
print("")

df_prices["strike_price"] = df_prices["strike_price"]/1000

######################################################################################
# Loading data on dividend yield and interest rates:
query = "select * from OPTIONM.IDXDVD where secid = _secid_".replace("_secid_", str(secid))
div_yield = db.raw_sql(query)

query = "select * from OPTIONM.ZEROCD where days < 365*2"
int_rates = db.raw_sql(query)

######################################################################################
# Merging dividend yield right away:
df_prices["date"] = pd.to_datetime(df_prices["date"])
df_prices["exdate"] = pd.to_datetime(df_prices["exdate"])
div_yield["date"] = pd.to_datetime(div_yield["date"])

df_prices = pd.merge(df_prices, div_yield.rename({"rate":"div_yield"}, axis = 1),
                     on = "date", how = "left")

######################################################################################
# Interpolating interest rates for each (date, exdate) pair:
# For every (date, exdate) need to find an interest rate:
df_date_exdate = df_prices[["date", "exdate"]].drop_duplicates()
df_date_exdate = df_date_exdate[df_date_exdate.date >= "1986-01-01"]

# Looping through pairs of date, exdate to find interest rate by
# lineary interpolating available libor rates:
rate_list = []
for i_row in range(df_date_exdate.shape[0]):
    if i_row % 1000 == 0:
        print("Row %d out of %d" % (i_row, df_date_exdate.shape[0]))
    row = df_date_exdate.iloc[i_row]
    date = row["date"]
    exdate = row["exdate"]
    int_rates_sub = int_rates[(int_rates.date == date) & (~int_rates.rate.isnull())]

    if int_rates_sub.shape[0] == 0:
        # If there is no data for this day, get the last available date:
        # 1. Get the last date available
        last_available_date = int_rates[(int_rates["date"] <= date) & (~int_rates["rate"].isnull())].iloc[-1]["date"]
        int_rates_sub = int_rates[int_rates.date == last_available_date]

    x = int_rates_sub["days"]
    y = int_rates_sub["rate"]

    days_to_maturity = (exdate - date).days
    
    rate = np.interp(days_to_maturity, x, y)
    rate_list.append(rate)

df_date_exdate["int_rate"] = rate_list

# Merging data on rates to option data:
df_prices = pd.merge(df_prices, df_date_exdate, on = ["date", "exdate"], how = "left")


######################################################################################
# Filters related to dividend yield and interest rates:
df_prices["T"] = [(x.days - 1)/365 for x in df_prices.exdate - df_prices.date]
df_prices["call_min_price"] = np.maximum(0, df_prices["spot"] - np.exp(-df_prices["int_rate"]*df_prices["T"])*df_prices["strike_price"] - (df_prices["div_yield"]/df_prices["int_rate"])*df_prices["spot"]*(1 - np.exp(-df_prices["int_rate"]* df_prices["T"])))
df_prices["put_min_price"] = np.maximum(0, np.exp(-df_prices["int_rate"]*df_prices["T"])*df_prices["strike_price"] + (df_prices["div_yield"]/df_prices["int_rate"])*df_prices["spot"]*(1 - np.exp(-df_prices["int_rate"]* df_prices["T"])) - df_prices["spot"])

df_prices = df_prices[((df_prices["price"] >= df_prices["call_min_price"]) & (df_prices.cp_flag == "C")) | 
                      ((df_prices["price"] >= df_prices["put_min_price"]) & (df_prices.cp_flag == "P"))]

df_prices.to_csv("data/raw_data/opt_data_spx_OM_ITM_OTM.csv", index = False)





query = "select distinct ticker, mgmt_cd, et_flag, inst_fund, retail_fund, open_to_inv from FUND_SUMMARY2"
mf = db.raw_sql(query)
mf







