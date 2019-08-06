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
from stargazer.stargazer import Stargazer

os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

# Reading data
path = "/Users/rsigalov/Documents/PhD/disaster-risk-revision/data/CME_data/"
file_list = os.listdir(path)

for i, file in enumerate(file_list):
    print(i)
    if i == 0:
        df = pd.read_csv(path + file)
    else:
        df_to_append = pd.read_csv(path + file)
        df = df.append(df_to_append)
 
####################################################################
# Cleaning up columns
####################################################################
volume_colnames = df.columns[["Volume" in x for x in df.columns]]

# replacing NaNs with zeros for volume:
for col in volume_colnames:
    df.loc[df[col].isnull(),col] = 0
    
df["volume"] = df[volume_colnames].sum(axis = 1)
df = df.drop(volume_colnames, axis = 1)
        
# For close price take "Floor Close Price":
df["price"] = df["Floor Close Price"]

# dropping all price related columns
price_colnames = df.columns[["Price" in x for x in df.columns]]
price_colnames = price_colnames[price_colnames != "Strike Price"]
df = df.drop(price_colnames, axis = 1)

# dropping all Bid/Ask indicators:
indicator_colnames = df.columns[["Indicator" in x for x in df.columns]]
df = df.drop(indicator_colnames, axis = 1)

# Dropping columns with zero nonmissing observations
df = df.drop(["Asset Class", "Clearing Code", "Underlying Product Code", "Delta", "Last Trade Date"], axis = 1)

# Removing extra columnsL
df = df.drop(["Exchange Code", "Product Code", "Product Description", "Product Type "], axis = 1)

####################################################################
# Dealing with expiration dates: for each specified year-month need
# to find the date of the third Friday
####################################################################

df_expiry = df[["Contract Year", "Contract Month"]].drop_duplicates()
third_friday_list = []
for i_row in range(df_expiry.shape[0]):
    # Getting year and month:
    year = df_expiry.iloc[i_row, 0]
    month = df_expiry.iloc[i_row, 1]
    
    # Getting first and last day of month:
    first_day = pd.to_datetime(str(year) + "-" + str(month) + "-01")
    last_day = first_day + pd.offsets.MonthEnd(0)
    
    # Looping through all days in a month and looking for the third
    first_day_weekday = first_day.dayofweek
    day_delta = (11 - first_day_weekday) % 7
    first_friday = first_day + pd.DateOffset(day_delta)
    third_friday = first_friday + pd.DateOffset(14)
    third_friday_list.append(third_friday)
    
df_expiry["third_friday"] = third_friday_list

# Merging third friday as the date of expiry for options:
df = pd.merge(df, df_expiry.rename({"third_friday":"exdate"}, axis = 1),
              on = ["Contract Year", "Contract Month"], how = "left")
df = df.rename({"Strike Price": "strike_price",
                "Trade Date": "date",
                "Put/Call ": "cp_flag",
                "Open Interest": "open_interest"}, axis = 1)
df["date"] = pd.to_datetime(df["date"], format = "%Y%m%d")
df["T"] = [(x.days-1)/365 for x in df["exdate"] - df["date"]]

# Removing unused columns:
df = df.drop(["Contract Year", "Contract Month", "Contract Day", "Settlement", 
              "Implied Volatility"], axis = 1)

# Getting data on S&P 500 index value:
spindx = pd.read_csv("data/spindx.csv")
spindx = spindx.rename({"caldt":"date", "spindx": "spot"}, axis = 1)
spindx["date"] = pd.to_datetime(spindx["date"], format = "%Y%m%d")
df = pd.merge(df, spindx, on = "date", how = "left")

#### Applying filters to old CME data
# Common filter for puts and calls:
df = df[df["open_interest"] > 0]

# Call specific filters
c_df = df[df["cp_flag"] == "C"]
c_df = c_df[c_df["strike_price"] >= c_df["spot"]] # OTM filter
c_df = c_df[c_df["price"] < c_df["spot"]]

# Put specific filters
p_df = df[df["cp_flag"] == "P"]
p_df = p_df[p_df["strike_price"] <= p_df["spot"]] # OTM filter
p_df = p_df[p_df["price"] < p_df["strike_price"]]
p_df = p_df[p_df["price"] >= np.maximum(0, p_df["strike_price"] - p_df["spot"])]

# Putting both back together:
df = c_df.append(p_df)
df = df.sort_values(["date", "exdate", "strike_price"])
df.to_csv("data/raw_data/opt_data_spx_all_CME.csv", index = False)
    
####################################################################
# Comparing LIBOR and Constant maturity rates for period 85 to 95
####################################################################
libor = pd.read_csv("data/libor_rates.txt", sep='\t')

# Replacing dots with NaNs for LIBOR rates:
libor = libor.replace({"USD12MD156N": {".": np.nan},
                       "USD1MTD156N": {".": np.nan},
                       "USD3MTD156N": {".": np.nan},
                       "USD6MTD156N": {".": np.nan}})

# Renaming variables in both datasets:
libor = libor.rename({"USD1MTD156N": "libor_1m",
                      "USD3MTD156N": "libor_3m",
                      "USD6MTD156N": "libor_6m",
                      "USD12MD156N": "libor_12m",
                      "DATE": "date"}, axis = 1)

libor["date"] = pd.to_datetime(libor["date"], format = "%Y-%m-%d")

########################################################
# Getting information about rates to options data
########################################################
# Converting rate to continuously compounded:
libor = np.log(libor.set_index("date").astype(float)/100+1).reset_index()

# Converting libor rates to a more suitable format for interpolation
libor = pd.melt(libor, id_vars = "date").replace({"variable": {"libor_1m": 30,
                                                       "libor_3m": 90,
                                                       "libor_6m": 180,
                                                       "libor_12m": 365}})
libor = libor.rename({"variable": "days", "value": "rate"}, axis = 1)
libor = libor.sort_values(["date", "days"])
libor["days"] = libor["days"].astype(float)  
libor["rate"] = libor["rate"].astype(float)
libor.to_csv("data/raw_data/libor_rates.csv", index = False)

# For every (date, exdate) need to find an interest rate:
df_date_exdate = df[["date", "exdate"]].drop_duplicates()
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
    libor_sub = libor[(libor.date == date) & (~libor.rate.isnull())]

    if libor_sub.shape[0] == 0:
        # If there is no data for this day, get the last available date:
        # 1. Get the last date available
        last_available_date = libor[(libor["date"] <= date) & (~libor["rate"].isnull())].iloc[-1]["date"]
        libor_sub = libor[libor.date == last_available_date]

    x = libor_sub["days"]
    y = libor_sub["rate"]

    days_to_maturity = (exdate - date).days
    
    rate = np.interp(days_to_maturity, x, y)
    rate_list.append(rate)

df_date_exdate["int_rate"] = rate_list

# Merging data on rates to option data:
df = pd.merge(df, df_date_exdate, on = ["date", "exdate"], how = "left")

########################################################
# Working with Shiller PD ratio
########################################################
# Preliminary data cleaning:
shiller_pd = pd.read_csv("data/shiller_pd.csv")
shiller_pd = shiller_pd.rename({"Date":"date"}, axis = 1)[["date", "P", "D"]]
shiller_pd["date"] = pd.date_range(start = "1871-01-01", freq = "M", periods = shiller_pd.shape[0])
shiller_pd["date"] = shiller_pd["date"] + pd.offsets.MonthBegin(-1)
#shiller_pd = shiller_pd[(shiller_pd["date"] >= "1986-01-01") & (shiller_pd["date"] < "1996-01-01")]
shiller_pd["rate"] = shiller_pd["D"]/shiller_pd["P"]
shiller_pd = shiller_pd[["date", "rate"]]

# Extending pd data for every day in each month:
shiller_pd["date_end_mon"] = shiller_pd["date"] + pd.offsets.MonthEnd(0)

for i_row in range(shiller_pd.shape[0]):
    date_start = shiller_pd["date"].iloc[i_row]
    date_end = shiller_pd["date_end_mon"].iloc[i_row]
    date_index = pd.date_range(start = date_start, end = date_end, freq = "D")
    
    if i_row == 0:
        div_yield = pd.DataFrame({"div_yield": [shiller_pd["rate"].iloc[i_row]] * len(date_index)})
        div_yield["date"] = date_index
    else:
        to_append = pd.DataFrame({"div_yield": [shiller_pd["rate"].iloc[i_row]] * len(date_index)})
        to_append["date"] = date_index
        div_yield = div_yield.append(to_append)
        
div_yield = div_yield[["date", "div_yield"]]
div_yield.to_csv("data/raw_data/div_yield_spx_old_CME.csv", index = False)
        
df = pd.merge(df, div_yield, on = ["date"], how = "left")
        
# Removing observations for the same date maturity:
df = df[df.date != df.exdate]
df = df[~df.int_rate.isnull()]

#df.to_csv("data/raw_data/opt_data_CME_ITM_OTM.csv", index = False)

df.to_csv("data/raw_data/opt_data_spx_all_CME.csv", index = False)






