"""
Created on Tue Jun  4 23:47:47 2019

@author: rsigalov
"""

from __future__ import print_function
from __future__ import division
import sys

import numpy as np
import pandas as pd

def main(argv = None):

    ########################################################
    # Combining disaster risk measures created with no
    # extrapolation
    ########################################################
    # Loading data on individual aggregated disaster measures for the
    # universe of common shares:
    vars_to_keeps = ["D_clamp", "D_in_sample", "rn_prob_20", "rn_prob_40", "rn_prob_60", "rn_prob_80"]
    print("Loading data on common share")
    print("")
    combined_disaster_cs_df = pd.read_csv("estimated_data/disaster-risk-series/agg_combined_crsp_cs_30days.csv")
    combined_disaster_cs_df = combined_disaster_cs_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_crsp_cs_60days.csv"))
    combined_disaster_cs_df = combined_disaster_cs_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_crsp_cs_90days.csv"))
    combined_disaster_cs_df = combined_disaster_cs_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_crsp_cs_120days.csv"))
    combined_disaster_cs_df = combined_disaster_cs_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_crsp_cs_150days.csv"))
    combined_disaster_cs_df = combined_disaster_cs_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_crsp_cs_180days.csv"))
    combined_disaster_cs_df = combined_disaster_cs_df[combined_disaster_cs_df["var"].isin(vars_to_keeps)]
    combined_disaster_cs_df["level"] = "crsp_cs"
    
    vars_to_keeps = ["D_clamp", "D_in_sample", "rn_prob_20", "rn_prob_40", "rn_prob_60", "rn_prob_80"]
    print("Loading data on common share")
    print("")
    combined_disaster_ucs_df = pd.read_csv("estimated_data/disaster-risk-series/agg_combined_union_cs_30days.csv")
    combined_disaster_ucs_df = combined_disaster_ucs_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_union_cs_60days.csv"))
    combined_disaster_ucs_df = combined_disaster_ucs_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_union_cs_90days.csv"))
    combined_disaster_ucs_df = combined_disaster_ucs_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_union_cs_120days.csv"))
    combined_disaster_ucs_df = combined_disaster_ucs_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_union_cs_150days.csv"))
    combined_disaster_ucs_df = combined_disaster_ucs_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_union_cs_180days.csv"))
    combined_disaster_ucs_df = combined_disaster_ucs_df[combined_disaster_ucs_df["var"].isin(vars_to_keeps)]
    combined_disaster_ucs_df["level"] = "union_cs"

    # Loading data on individial aggregated disaster measures for the
    # whole universe:
    print("Loading data on all SECIDs")
    print("")
    combined_disaster_all_df = pd.read_csv("estimated_data/disaster-risk-series/agg_combined_comb_30days.csv")
    combined_disaster_all_df = combined_disaster_all_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_comb_all_60days.csv"))
    combined_disaster_all_df = combined_disaster_all_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_comb_all_90days.csv"))
    combined_disaster_all_df = combined_disaster_all_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_comb_all_120days.csv"))
    combined_disaster_all_df = combined_disaster_all_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_comb_all_150days.csv"))
    combined_disaster_all_df = combined_disaster_all_df.append(pd.read_csv("estimated_data/disaster-risk-series/agg_combined_comb_all_180days.csv"))
    combined_disaster_all_df["level"] = "ind_all"

    combined_disaster_df = combined_disaster_cs_df.append(combined_disaster_all_df)
    combined_disaster_df = combined_disaster_df.append(combined_disaster_ucs_df)

    ################################################################
    # Combining data on SPX
    ################################################################
    # First loading OptionMetrics based data on SPX (short sample post Jan-96)
    print("Loading data on OptionMetrics SPX")
    print("")
    for i_days, days in enumerate([30, 40, 60, 100, 120, 180]):
        if i_days == 0:
            spx_OM = pd.read_csv("estimated_data/interpolated_D/int_D_spx_days_" + str(days) +".csv")
            spx_OM["days"] = days
        else:
            to_append = pd.read_csv("estimated_data/interpolated_D/int_D_spx_days_" + str(days) +".csv")
            to_append["days"] = days
            spx_OM = spx_OM.append(to_append)
                    
    spx_OM = spx_OM.sort_values(["date", "days"])
    spx_OM.to_csv("estimated_data/disaster-risk-series/spx_OM_daily_disaster.csv", index = False)

    # Next loading CBOE data based data on SPX (long sample post Jan-86):
    print("Loading data on CME SPX")
    print("")
    for i_days, days in enumerate([30, 40, 60, 100, 120, 180]):
        if i_days == 0:
            spx_CME = pd.read_csv("estimated_data/interpolated_D/int_D_spx_all_CME_days_" + str(days) +".csv")
            spx_CME["days"] = days
        else:
            to_append = pd.read_csv("estimated_data/interpolated_D/int_D_spx_all_CME_days_" + str(days) +".csv")
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
    # spx_OM_melt["extrapolation"] = "N"

    spx_CME_melt = pd.melt(spx_mon_CME.drop("secid", axis = 1), id_vars = ["date", "days"])
    spx_CME_melt = spx_CME_melt.rename({"variable": "var"}, axis = 1)
    spx_CME_melt["level"] = "sp_500_CME"
    spx_CME_melt["agg_type"] = "mean_all"
    # spx_CME_melt["extrapolation"] = "N"

    combined_disaster_df = combined_disaster_df.append(spx_OM_melt)
    combined_disaster_df = combined_disaster_df.append(spx_CME_melt)
    combined_disaster_df["date"] = pd.to_datetime(combined_disaster_df["date"])

    combined_disaster_df.to_csv("estimated_data/disaster-risk-series/combined_disaster_df.csv", index = False)

if __name__ == "__main__":
    sys.exit(main(sys.argv))




