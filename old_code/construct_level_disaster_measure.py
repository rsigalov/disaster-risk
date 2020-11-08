"""
This scripts takes interpolated series corresponding to union OM and CRSP
definition of Common Shares and constructs:

    (1) For each (secid, month, days) average disaster measure if there
    are at least 5 observations. This is needed to form portfolios on
    individual disaster measure
    (2) For each (secid, month) -//- 10 observations
    (3) For each month average disaster measure as a truncated average
    across all companies and maturities with at least 10 observations
"""
from __future__ import print_function
from __future__ import division
import sys

import numpy as np
import pandas as pd

def mean_with_truncation(x):
    return np.nanmean(x[(x <= np.nanquantile(x, 0.975)) & (x >= np.nanquantile(x, 0.025))])

def main(argv = None):

    VARIABLE = "D_clamp"

    # Loading interpolated data for all maturities and combining
    # them to construct a level measure:
    int_d_columns = ["secid", "date"] + [VARIABLE]
    int_d = pd.DataFrame(columns = int_d_columns)

    for days in [30, 60, 90, 120, 150, 180]:
        print(days)
        int_d_tmp = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_union_cs_" + str(days) + ".csv")
        int_d_tmp = int_d_tmp[int_d_columns]
        int_d_tmp["days"] = days
        int_d = int_d.append(int_d_tmp)

    int_d["date"] = pd.to_datetime(int_d["date"])
    int_d["date_mon"] = int_d["date"] + pd.offsets.MonthEnd(0)

    # Constructing monthly (secid, days) disaster measures
    mean_cnt = int_d.groupby(["secid", "date_mon", "days"])["date"].count().reset_index()
    mean_cnt = mean_cnt.rename(columns = {"date": "cnt"})
    mean_cnt = mean_cnt[mean_cnt.cnt >= 5]
    int_d_for_mon_ind_measure = pd.merge(
        mean_cnt, int_d, on = ["secid", "date_mon", "days"], how = "left")
    ind_mon_measure = int_d_for_mon_ind_measure.groupby(["secid", "date_mon", "days"])[VARIABLE].mean()
    ind_mon_measure.reset_index().to_csv("estimated_data/interpolated_D/mon_ind_disaster_days.csv", index = False)

    # Constructing monthly level secid disaster measure
    mean_cnt = int_d.groupby(["secid", "date_mon"])["date"].count().reset_index()
    mean_cnt = mean_cnt.rename(columns = {"date": "cnt"})
    mean_cnt = mean_cnt[mean_cnt.cnt >= 10]
    int_d_for_mon_ind_measure = pd.merge(
        mean_cnt, int_d, on = ["secid", "date_mon"], how = "left")
    ind_mon_measure = int_d_for_mon_ind_measure.groupby(["secid", "date_mon"])[VARIABLE].mean()
    ind_mon_measure.reset_index().to_csv("estimated_data/interpolated_D/mon_ind_disaster_level.csv", index = False)

    # Finally, constructing a general level factor:
    ind_mon_measure = int_d_for_mon_ind_measure.groupby(["date_mon"])[VARIABLE].apply(mean_with_truncation)
    ind_mon_measure.reset_index().to_csv("estimated_data/interpolated_D/level_disaster_series.csv", index = False)

if __name__ == "__main__": sys.exit(main(sys.argv))
