"""
This script looks into the industry composition of interpolated
disaster measures
"""

import pandas as pd
import numpy as np

# Loading individual disaster measures:
int_d = pd.DataFrame(columns = ["secid", "date", "D_clamp"])

for days in [30, 60, 90, 120, 150, 180]:
    print(days)
    int_d_to_append = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_union_cs_" + str(days) + ".csv")
    int_d_to_append = int_d_to_append[["secid", "date", "D_clamp"]]
    int_d_to_append["days"] = days
    int_d = int_d.append(int_d_to_append)

int_d["date"] = pd.to_datetime(int_d["date"])
int_d["date_mon"] = int_d["date"] + pd.offsets.MonthEnd(0)

# Loading WRDS OM/CRSP linking table
om_crsp = pd.read_csv("om_crsp_wrds_linking_table.csv").rename(columns = {"PERMNO":"permno"})
om_crsp["sdate"] = pd.to_datetime(om_crsp["sdate"], format = "%Y%m%d")
om_crsp["edate"] = pd.to_datetime(om_crsp["edate"], format = "%Y%m%d")

# For each (secid, date) getting the best link from OM-CRSP linking table:
df_crsp_merge = pd.merge(int_d[["secid", "date"]].drop_duplicates(), om_crsp, on = "secid", how = "left")
df_crsp_merge = df_crsp_merge[
    (df_crsp_merge.date <= df_crsp_merge.edate) &
    (df_crsp_merge.date >= df_crsp_merge.sdate)]
df_crsp_merge = df_crsp_merge.drop(columns = ["sdate", "edate", "score"])

# Loading data on CRSP SIC codes:
crsp_sic = pd.read_csv("data/crsp_sic_codes.csv")
crsp_sic["SICCD"] = pd.to_numeric(crsp_sic["SICCD"], errors = "coerce")
crsp_sic["date"] = pd.to_datetime(crsp_sic["date"], format = "%Y%m%d")
df_crsp_merge["date_mon"] = df_crsp_merge["date"] + pd.offsets.MonthEnd(0)
df_crsp_merge = pd.merge(df_crsp_merge, crsp_sic, left_on = ["permno", "date_mon"],
                         right_on = ["PERMNO", "date"], how = "left")
df_crsp_merge = df_crsp_merge.drop(columns = ["date_mon", "date_y"]).rename(columns = {"date_x":"date"})

# loading data on 12 FF industries 
ff_ind = pd.read_csv("ff_12_ind.csv")
df_crsp_merge["ff_ind"] = None
for i_row in range(ff_ind.shape[0]):
    print(i_row)
    beg_sic = ff_ind.iloc[i_row,2]
    end_sic = ff_ind.iloc[i_row,3]
    ff_ind_name = ff_ind.iloc[i_row,1]
    df_crsp_merge.loc[
            (df_crsp_merge.SICCD >= beg_sic) & 
            (df_crsp_merge.SICCD <= end_sic), "ff_ind"] = ff_ind_name

df_crsp_merge.loc[df_crsp_merge.SICCD.notnull() & df_crsp_merge.ff_ind.isnull(), "ff_ind"] = "Other"
df_crsp_merge = df_crsp_merge[df_crsp_merge.ff_ind.notnull()]
df_crsp_merge = df_crsp_merge[["secid", "date", "SICCD", "ff_ind"]]

# Merging back interpolated series:
int_d = pd.merge(
        int_d, df_crsp_merge,
        on = ["date", "secid"], how = "left")

# Averaging disaster series within industries on daily and monthly levels:
def mean_with_truncation(x):
    return np.mean(x[(x <= np.quantile(x, 0.975)) & (x >= np.quantile(x, 0.025))])

int_d = int_d[int_d.D_clamp.notnull()]
ff_ind_disaster = int_d.groupby(["date", "ff_ind"])["D_clamp"].apply(mean_with_truncation).rename("D_clamp")
ff_ind_disaster = pd.pivot_table(pd.DataFrame(ff_ind_disaster), columns = "ff_ind", index = "date")
ff_ind_disaster.plot(figsize = (10,8))

# Constructing monthly measure:
mean_cnt_df = mean_cnt_df[mean_cnt_df.cnt >= 10]
df_filter = pd.merge(mean_cnt_df, df, on = ["secid", agg_level_var], how = "left")
df_mean_mon = df_filter.groupby(["secid", agg_level_var])[agg_var].mean().reset_index()
level = df_mean_mon.groupby([agg_level_var])[agg_var].apply(mean_with_truncation).rename(agg_var)
num_comps = df_mean_mon.groupby([agg_level_var])["secid"].count()
