"""
Created on Tue Jun  4 23:47:47 2019

@author: rsigalov
"""

import numpy as np
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance
from matplotlib import pyplot as plt
import matplotlib as mpl
import os 
os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

########################################################
# Combining disaster risk measures created with no
# extrapolation
########################################################
# Loading data on individual aggregated disaster measures:
combined_disaster_df = pd.read_csv("estimated_data/disaster-risk-series/agg_combined_30days.csv")
combined_disaster_df = combined_disaster_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_60days.csv"))
combined_disaster_df = combined_disaster_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_120days.csv"))
combined_disaster_df = combined_disaster_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_180days.csv"))
combined_disaster_df["date"] = pd.to_datetime(combined_disaster_df["date"])

# Replacing rn_prob_X_ann with rn_prob_X_mon:
combined_disaster_df = combined_disaster_df.replace({
        "var": {"rn_prob_20ann": "rn_prob_20mon", 
                "rn_prob_40ann": "rn_prob_40mon"}})

combined_disaster_df["extrapolation"] = "N"

################################################################
# Combining data on SPX
################################################################
# First loading OptionMetrics based data on SPX (short sample post Jan-96)
for i_days, days in enumerate([30, 40, 60, 100, 120, 180]):
    if i_days == 0:
        spx_OM = pd.read_csv("estimated_data/disaster-risk-series/int_D_spx_days_" + str(days) +".csv")
        spx_OM["days"] = days
    else:
        to_append = pd.read_csv("estimated_data/disaster-risk-series/int_D_spx_days_" + str(days) +".csv")
        to_append["days"] = days
        spx_OM = spx_OM.append(to_append)
                
spx_OM = spx_OM.sort_values(["date", "days"])
spx_OM.to_csv("estimated_data/disaster-risk-series/spx_OM_daily_disaster.csv", index = False)

# Next loading CBOE data based data on SPX (long sample post Jan-86):
for i_days, days in enumerate([30, 40, 60, 100, 120, 180]):
    if i_days == 0:
        spx_CME = pd.read_csv("estimated_data/disaster-risk-series/int_D_spx_all_CME_days_" + str(days) +".csv")
        spx_CME["days"] = days
    else:
        to_append = pd.read_csv("estimated_data/disaster-risk-series/int_D_spx_all_CME_days_" + str(days) +".csv")
        to_append["days"] = days
        spx_CME = spx_CME.append(to_append)

spx_CME = spx_CME.sort_values(["date", "days"])
spx_CME.to_csv("estimated_data/disaster-risk-series/spx_CME_daily_disaster.csv", index = False)

# Averaging both of them on monthly level:
def aggregate_at_monthly_level(spx):
    spx["date"] = pd.to_datetime(spx["date"])
    spx["date_mon"] = spx["date"] + pd.offsets.MonthEnd(0)
    colnames = list(spx.columns)
    colnames = [x for x in colnames if x not in ["date", "date_mon", "days"]]
    spx_mon = spx.drop(["date"], axis = 1).groupby(["date_mon", "days"]).apply(np.nanmean, axis = 0).apply(pd.Series)
    spx_mon.columns = colnames
    spx_mon = spx_mon.reset_index()
    spx_mon = spx_mon.rename({"date_mon": "date"}, axis = 1)
    return spx_mon

spx_mon_OM = aggregate_at_monthly_level(spx_OM)
spx_mon_OM.to_csv("estimated_data/disaster-risk-series/spx_OM_monthly_disaster.csv", index = False)

spx_mon_CME = aggregate_at_monthly_level(spx_CME)
spx_mon_CME.to_csv("estimated_data/disaster-risk-series/spx_CME_monthly_disaster.csv", index = False)

# Adding to other disaster measures:
spx_OM_melt = pd.melt(spx_mon_OM.drop("secid", axis = 1), id_vars = ["date", "days"])
spx_OM_melt = spx_OM_melt.rename({"variable": "var"}, axis = 1)
spx_OM_melt["level"] = "sp_500_OM"
spx_OM_melt["agg_type"] = "mean_all"
spx_OM_melt["extrapolation"] = "N"

spx_CME_melt = pd.melt(spx_mon_CME.drop("secid", axis = 1), id_vars = ["date", "days"])
spx_CME_melt = spx_CME_melt.rename({"variable": "var"}, axis = 1)
spx_CME_melt["level"] = "sp_500_CME"
spx_CME_melt["agg_type"] = "mean_all"
spx_CME_melt["extrapolation"] = "N"

combined_disaster_df = combined_disaster_df.append(spx_OM_melt)
combined_disaster_df = combined_disaster_df.append(spx_CME_melt)

combined_disaster_df.to_csv("estimated_data/disaster-risk-series/combined_disaster_df.csv", index = False)






