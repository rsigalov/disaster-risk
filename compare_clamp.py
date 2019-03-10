#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare estimation results for clamped vs. extrapolated SVi curves:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
from sklearn.decomposition import PCA
import os
os.chdir('/Users/rsigalov/Documents/PhD/disaster-risk-revision')

# Apple from extrapolated SVI volatility smile:
df_ext = pd.read_csv("output/var_ests_equity_1.csv")
df_ext = df_ext[df_ext["secid"] == 101594]

df_clamp = pd.read_csv("aapl_2017_clamp_estimates.csv")


############################################################
# Estimate D's for extrapolated SVI
############################################################
def calc_Ds(df, T):

    measure_list = ["", "_in_sample", "_5_5", "_otm_in_sample", "_otm_5_5", "_otm1_in_sample"]
    
    for i_measure in range(len(measure_list)):
        print(i_measure)
        df_short = df[["secid", "date", "T", "V" + measure_list[i_measure], 
                       "IV" + measure_list[i_measure]]]
        df_short["D"] = df_short["V" + measure_list[i_measure]] - df_short["IV" + measure_list[i_measure]]
        df_short = df_short.drop(["V" + measure_list[i_measure], "IV" + measure_list[i_measure]], axis = 1)
        
        secid_list = []
        date_list = []
        D_list = []
        
        for (name, sub_df) in df_short.groupby(["secid", "date"]):                
            secid_list.append(name[0])
            date_list.append(name[1])
            
            D_list.append(np.interp(T, sub_df["T"], sub_df["D"]))
            
#            # Removing NaNs for interpolation function:
#            x = sub_df["T"]
#            y = sub_df["D"]
#            stacked = np.vstack((x.T, y.T)).T
#    
#            stacked_filter = stacked[~np.isnan(stacked).any(axis = 1)]
#            
#            x_new = stacked_filter[:,0]
#            y_new = stacked_filter[:,1]
#            
#            # Interpolating. Since we removed NaN if can't interpolate the function
#            # will simply return NaNs
#            if len(x_new) == 0:
#                D_list.append(np.nan)
#            else:
#                D_list.append(np.interp(T, x_new, y_new))
        
        D_df_to_merge = pd.DataFrame({"secid":secid_list, "date":date_list, "D":D_list})
        D_df_to_merge = D_df_to_merge.sort_values(["secid", "date"])
        
        # Getting the first date of the month average the series:
        D_df_to_merge = D_df_to_merge.replace([np.inf, -np.inf], np.nan)
        D_df_to_merge.columns = ["secid", "date", "D" + measure_list[i_measure]]
        
        if i_measure == 0:
            D_df = D_df_to_merge
        else:
            D_df = pd.merge(D_df, D_df_to_merge, on = ["secid", "date"])
            
    return D_df
    


D_ext = calc_Ds(df_ext, 30/365)
D_clamp = calc_Ds(df_clamp, 30/365)


def plot_missing_positive_shares(D_df):

    measure_list = ["", "_in_sample", "_5_5", "_otm_in_sample", "_otm_5_5", "_otm1_in_sample"]
    for (i_measure, measure) in enumerate(measure_list):
        print(i_measure)
        cnt_all = D_df.groupby("date")["D" + measure].size().reset_index().rename(columns={"D" + measure: "cnt_all"})
        cnt_not_null = D_df.groupby("date")["D" + measure].count().reset_index().rename(columns={"D" + measure: "cnt_not_null"})
        cnt_pos = D_df[D_df["D" + measure] > 0].groupby("date")["D" + measure].count().reset_index().rename(columns={"D" + measure: "cnt_pos"})
        
        cnt_merged = pd.merge(cnt_all, cnt_not_null)
        cnt_merged = pd.merge(cnt_merged, cnt_pos)
        
        cnt_merged["date"] = pd.to_datetime(cnt_merged["date"])
        
        # Calculating shares:
        cnt_merged["s_not_null" + measure] = cnt_merged["cnt_not_null"]/cnt_merged["cnt_all"]
        cnt_merged["s_pos" + measure] = cnt_merged["cnt_pos"]/cnt_merged["cnt_all"]
        
        # Averaging within a month:
        cnt_merged["date_trunc"] = cnt_merged["date"] - pd.offsets.MonthBegin(1)
        cnt_mon = cnt_merged.groupby("date_trunc")[["s_not_null" + measure, "s_pos" + measure]].mean()
        
        if i_measure == 0:
            shares_all = cnt_merged[["date", "cnt_all", "s_not_null" + measure, "s_pos"  + measure]]
        else:
            shares_all = pd.merge(shares_all, cnt_merged[["date", "s_not_null" + measure, "s_pos"  + measure]], on = "date")
            
        if i_measure == 0:
            shares_all_mon = cnt_mon
        else:
            shares_all_mon = pd.merge(shares_all_mon, cnt_mon, on = "date_trunc")
            
    
    
    shares_all_mon[["s_not_null" + x for x in measure_list]].plot(figsize = (10,8))
    shares_all_mon[["s_pos" + x for x in measure_list]].plot(figsize = (10,8))

D_ext["date"] = pd.to_datetime(D_ext["date"])
D_clamp["date"] = pd.to_datetime(D_clamp["date"])

D_ext.set_index("date").drop("secid", axis=1).plot()

D_clamp.set_index("date").drop("secid", axis=1).plot()

plt.plot(D_clamp["date"], D_clamp["D"])
plt.plot(D_ext["date"], D_ext["D_in_sample"])
plt.legend()








