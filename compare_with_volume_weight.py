#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:50:22 2019

@author: rsigalov
"""

import crsp_comp
import rolling_disaster_betas



int_d = pd.DataFrame(columns = ["secid", "date", "D_clamp"])

for days in [30, 60, 90, 120, 150, 180]:
    print(days)
    int_d_to_append = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_union_cs_" + str(days) + ".csv")
    int_d_to_append = int_d_to_append[["secid", "date", "D_clamp"]]
    int_d_to_append["days"] = days
    int_d = int_d.append(int_d_to_append)

int_d["date"] = pd.to_datetime(int_d["date"])
int_d["date_mon"] = int_d["date"] + pd.offsets.MonthEnd(0)

mean_cnt_df = int_d.groupby(["secid", "date_mon"])["D_clamp"].count().reset_index()
mean_cnt_df = mean_cnt_df.rename({"D_clamp": "cnt"}, axis = 1)
mean_cnt_df = mean_cnt_df[mean_cnt_df.cnt >= 10]
int_d_filter = pd.merge(mean_cnt_df, int_d, on = ["secid", "date_mon"], how = "left")
int_d_mean_mon = int_d_filter.groupby(["secid", "date_mon"])["D_clamp"].mean().reset_index()

vol = pd.read_csv("estimated_data/option_volume/option_volume_1.csv")
vol = vol.append(pd.read_csv("estimated_data/option_volume/option_volume_2.csv"))
vol = vol.append(pd.read_csv("estimated_data/option_volume/option_volume_3.csv"))
vol.rename(columns = {"open_inetrest":"open_interest"}, inplace = True)
vol["obs_date"] = pd.to_datetime(vol["obs_date"])

vol_daily_mean = vol.groupby(["secid", "obs_date"])["volume","open_interest"].mean().reset_index()
vol_daily_mean["date_mon"] = vol_daily_mean["obs_date"] + pd.offsets.MonthEnd(0)
vol_mon_mean = vol_daily_mean.groupby(["secid", "date_mon"])["volume","open_interest"].mean().reset_index()




# Merging with volume and open_interest values:
int_d_mean_mon = pd.merge(int_d_mean_mon, vol_mon_mean, on = ["date_mon", "secid"], how = "left")

def mean_with_truncation(x):
    return np.mean(x[(x <= np.quantile(x, 0.99)) & (x >= np.quantile(x, 0.01))])

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    ind_not_extreme = (d <= np.quantile(d, 0.99)) & (d >= np.quantile(d, 0.01))
    d = d[ind_not_extreme]
    w = w[ind_not_extreme]

    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

level_op_int_w = int_d_mean_mon.groupby(["date_mon"]).apply(lambda x: wavg(x, "D_clamp", "open_interest"))
level_volume_w = int_d_mean_mon.groupby(["date_mon"]).apply(lambda x: wavg(x, "D_clamp", "volume"))
level = int_d_mean_mon.groupby(["date_mon"])["D_clamp"].apply(mean_with_truncation)

level.plot(figsize = (10,7), label = "Simple mean")
level_volume_w.plot(label = "Volume weighted")
level_op_int_w.plot(label = "open_interest weighted")
plt.legend()

level_df = pd.merge(level.rename("mean"), level_volume_w.rename("vol_weight"),
                    left_index = True, right_index = True)
level_df = pd.merge(level_df, level_op_int_w.rename("op_int_weight"),
                    left_index = True, right_index = True)

factors = level_df.diff()

level_df.diff().corr().round(4)
level_df.plot()


df = pd.read_csv("estimated_data/disaster_sorts/reg_results_agg_volume.csv")

df[(df.FF == 0) & (df.port == "ew_1")][["port", "variable", "alpha", "alpha_se"]]
df[(df.FF == 0) & (df.port == "ew_5")][["port", "variable", "alpha", "alpha_se"]]
df[(df.FF == 0) & (df.port == "ew_diff")][["port", "variable", "alpha", "alpha_se"]]

df[(df.FF == 0) & (df.port == "vw_1")][["port", "variable", "alpha", "alpha_se"]]
df[(df.FF == 0) & (df.port == "vw_5")][["port", "variable", "alpha", "alpha_se"]]
df[(df.FF == 0) & (df.port == "vw_diff")][["port", "variable", "alpha", "alpha_se"]]

factors = pd.read_csv("estimated_data/disaster-risk-series/volume_weight_factors.csv")






