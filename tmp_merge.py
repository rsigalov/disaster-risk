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

df1 = pd.read_csv("estimated_data/V_IV/var_ests_spx1.csv")
df2 = pd.read_csv("estimated_data/V_IV/var_ests_spx2.csv")

df1 = df1[["secid", "date", "T", "V", "IV", "V_in_sample", "IV_in_sample",
           "V_clamp", "IV_clamp","rn_prob_2sigma"]]

df = pd.merge(df1, df2, on = ["secid", "date", "T"], how = "inner")

df.to_csv("estimated_data/V_IV/var_ests_spx.csv", index = False)



df = pd.read_csv("estimated_data/interpolated_D/int_D_spx_days_30.csv")
df = df.sort_values("date")
df["date"] = pd.to_datetime(df["date"])
df["date_adj"] = df["date"] + pd.offsets.MonthEnd(0)
df_agg = df.groupby("date_adj").mean()

df_agg[["rn_prob_5mon","rn_prob_10mon","rn_prob_15mon","rn_prob_20mon"]].plot()
df_agg[["D_in_sample", "D_clamp"]].plot()

df_agg[["D_clamp", "rn_prob_15mon"]].plot()


