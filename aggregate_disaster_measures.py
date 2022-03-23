"""
This script aggregates disaster measures. It reads interpolated disaster
measures from the folder estimated_data/interpolated_D/ applies filters
(e.g. requiring a minimum of 10 observations per company-month) and then
averages individual disaster measures with 1% truncation on bottom and top
"""

from __future__ import print_function
from __future__ import division
import sys

import numpy as np
import pandas as pd
from functools import reduce
# import os
# os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")

def mean_with_truncation(x):
    return np.mean(x[(x <= np.quantile(x, 0.99)) & (x >= np.quantile(x, 0.01))])
    
def aggregate_disaster_at_level(df, agg_var, agg_level_var, min_obs_agg_level):
    mean_cnt_df = df.groupby(["secid", agg_level_var])["value"]\
                    .count()\
                    .reset_index()\
                    .rename({"value": "cnt"}, axis = 1)
                       
    mean_cnt_df = mean_cnt_df[mean_cnt_df.cnt >= 10]
    df_filter = pd.merge(mean_cnt_df, df, on = ["secid", agg_level_var], how = "left")
    df_mean_mon = df_filter.groupby(["secid", agg_level_var])["value"].mean().reset_index()
    level = df_mean_mon.groupby([agg_level_var])["value"].apply(mean_with_truncation).rename("value")
    num_comps = df_mean_mon.groupby([agg_level_var])["secid"].count()
    
    level.index.names = ["date"]
    num_comps.index.names = ["date"]
    
    return level, num_comps

class aggregated_disaster_measure():
    def __init__(self, df, level, maturity, variable, agg_freq, min_comps_freq):
        self.maturity = maturity
        self.variable = variable
        self.agg_freq = agg_freq
        self.min_comps_freq = min_comps_freq
        self.level = level

        if maturity == "level":
            self.value, self.num_comps = aggregate_disaster_at_level(
                    df[(df["variable"] == variable)], variable, agg_freq, min_comps_freq)
        else:
            self.value, self.num_comps = aggregate_disaster_at_level(
                    df[(df["days"] == maturity) & (df["variable"] == variable)], variable, agg_freq, min_comps_freq)
            
        self.value.rename("value", inplace = True)
        self.num_comps.rename("num_comps", inplace = True)
            
    def output_df(self):
        df = pd.merge(self.value, self.num_comps, left_index = True, right_index = True)
        df["level"] = self.level
        df["maturity"] = self.maturity
        df["variable"] = self.variable
        df["agg_freq"] = self.agg_freq
        df["min_comps_freq"] = self.min_comps_freq
        
        return df
        
def main(argv = None):
    # Loading all interpolated individual disaster measures for all
    # maturities: 30 days, 60 days, ..., 180 days
    print("")
    print("---- Loading interpolated disaster measures ----")
    print("")

    # Individual measures
    int_ind_list = []
    for days in [30, 60, 90, 120, 150, 180]:
        int_ind = pd.read_csv(f"data/interpolated_D/interpolated_disaster_individual_{days}.csv")
        int_ind["date"] = pd.to_datetime(int_ind["date"])
        int_ind["date_mon"] = int_ind["date"] + pd.offsets.MonthEnd(0)
        int_ind["date_week"] = int_ind['date'] - int_ind['date'].dt.weekday.astype('timedelta64[D]')
        int_ind_list.append(int_ind)

    int_ind = pd.concat(int_ind_list, ignore_index=True)
    del(int_ind_list)

    # spx measures
    int_spx_list = []
    for days in [30, 60, 90, 120, 150, 180]:
        int_spx = pd.read_csv(f"data/interpolated_D/interpolated_disaster_spx_{days}.csv")
        int_spx["date"] = pd.to_datetime(int_spx["date"])
        int_spx["date_mon"] = int_spx["date"] + pd.offsets.MonthEnd(0)
        int_spx["date_week"] = int_spx['date'] - int_spx['date'].dt.weekday.astype('timedelta64[D]')
        int_spx_list.append(int_spx)

    int_spx = pd.concat(int_spx_list, ignore_index=True)
    del(int_spx_list)
    
    # Constructing level disaster measures:
    # Calculating level component on monthly level
    print("\n---- Aggregating Individual Disaster Measures ----\n")
    
    disaster_measure_list = []
    
    for variable in ["D_clamp", "rn_prob_20", "rn_prob_80"]:
        for maturity in ["level", 30, 60, 90, 120, 150, 180]:
            print("Individual measure: %s, %s" % (variable, maturity))
            if maturity == "level":
                min_comps = 15
            else:
                min_comps = 5
                
            disaster_measure_list.append(
                    aggregated_disaster_measure(
                            int_ind, "Ind", maturity, variable, "date_mon", min_comps))
            
    print("\n---- Aggregating SPX Disaster Measures ----\n")
    for variable in ["D_clamp", "rn_prob_5", "rn_prob_20"]:
        for maturity in ["level", 30, 60, 90, 120, 150, 180]:
            print("SPX measure: %s, %s" % (variable, maturity))
            disaster_measure_list.append(
                    aggregated_disaster_measure(int_spx, "SPX", maturity, variable, "date_mon", 1))
    
    print("\n---- Aggregating weekly level series ----\n")
    for variable in ["D_clamp", "rn_prob_20", "rn_prob_80"]:
        for maturity in ["level", 30, 60, 90, 120, 150, 180]:
            print("Individual measure: %s, %s" % (variable, maturity))
            disaster_measure_list.append(
                    aggregated_disaster_measure(int_ind, "Ind", "level", variable, "date_week", 5))
        
    for variable in ["D_clamp", "rn_prob_5", "rn_prob_20"]:
        for maturity in ["level", 30, 60, 90, 120, 150, 180]:
            print("SPX measure: %s, %s" % (variable, maturity))
            disaster_measure_list.append(
                    aggregated_disaster_measure(int_spx, "SPX", "level", variable, "date_week", 1))
        
    # Combining and saving measures:    
    output_df = reduce(lambda df1, df2: df1.append(df2), 
                       [x.output_df() for x in disaster_measure_list])
    
    output_df.to_csv(f"data/disaster_risk_measures/disaster_risk_measures.csv")
 
if __name__ == "__main__": 
    sys.exit(main(sys.argv))
    