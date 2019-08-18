#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
companies that were downloaded in the first pass, (2) including the missing
companies that I estimated later and (3) including only missing companies
that are common stocks as matched with CRSP
"""
import pandas as pd
import numpy as np
from functools import reduce
import wrds

# Loading not-filtered data on interpolated D and looking for overlap in
# (secid,date). Take the new estimation for these observations:
days = 120
file_name = "estimated_data/interpolated_D/int_ind_disaster_days_" + str(days) + ".csv"
old_int_d = pd.read_csv(file_name) 
old_int_d["date"] = pd.to_datetime(old_int_d["date"])
old_int_d = old_int_d[["secid", "date", 'D_clamp', 'D_in_sample',"rn_prob_20","rn_prob_40",
                       "rn_prob_60","rn_prob_80"]]

file_name = "estimated_data/interpolated_D/int_ind_disaster_missing_days_" + str(days) + ".csv"
missing_int_d = pd.read_csv(file_name) 
missing_int_d["date"] = pd.to_datetime(missing_int_d["date"])
missing_int_d["flag_overlap"] = 1

df_index = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_index_days_" + str(days)+ ".csv")
df_index["date"] = pd.to_datetime(df_index["date"])

unique_old_obs = pd.merge(old_int_d[["secid", "date"]],
                          missing_int_d[["secid","date","flag_overlap"]],
                          on = ["secid", "date"], how = "left")
unique_old_obs = unique_old_obs[unique_old_obs.flag_overlap != 1]
unique_old_obs = pd.merge(unique_old_obs[["secid", "date"]], old_int_d, 
                          on = ["secid", "date"], how = "left")

new_old_obs = pd.merge(old_int_d[["secid", "date"]],
                       missing_int_d[["secid","date","flag_overlap"]],
                       on = ["secid", "date"], how = "left")
new_old_obs = new_old_obs[new_old_obs.flag_overlap == 1]
new_old_obs = pd.merge(new_old_obs[["secid","date"]], missing_int_d.drop(columns = "flag_overlap"),
                       on = ["secid", "date"], how = "left")

# Combining the two together:
old_int_d_cleaned = unique_old_obs.append(new_old_obs)

# Now removing overlapped observations from missing_int_df:
old_int_d["flag_overlap"] = 1
not_overlap_new_obs = pd.merge(missing_int_d[["secid","date"]], 
                               old_int_d[["secid", "date", "flag_overlap"]],
                               on = ["secid", "date"], how = "left")
not_overlap_new_obs = not_overlap_new_obs[not_overlap_new_obs.flag_overlap != 1]
not_overlap_new_obs = pd.merge(not_overlap_new_obs[["secid", "date"]],
                               missing_int_d, on = ["secid","date"], how = "left")

old_int_d_cleaned["old_obs"] = 1
not_overlap_new_obs["old_obs"] = 0
int_d_combined = old_int_d_cleaned.append(not_overlap_new_obs)

# Saving the result separately
int_d_combined = int_d_combined.drop(columns = "flag_overlap")


########################################################
# Now looking through a particular measure and compare
# the aggregated disaster measure before and after we
# add new observation
########################################################
db = wrds.Connection()

# First, loading data on match of SECID and CRSP:
query = """
select secid, cusip, issue_type
from OPTIONM.SECURD
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df_issue_type = db.raw_sql(query)

query = """
select 
    cusip, ncusip, namedt, nameendt, shrcd, exchcd, ticker
from crsp.msenames
"""
query = query.replace('\n', ' ').replace('\t', ' ')
df_crsp_cusip = db.raw_sql(query, date_cols = ["namedt", "nameendt"])

#####
int_d_combined = pd.merge(int_d_combined, df_issue_type, on = "secid", how = "left")

crsp_matched = pd.merge(
        int_d_combined, df_crsp_cusip.drop(columns = "cusip"), 
        left_on = "cusip", right_on = "ncusip", how = "left")
crsp_matched = crsp_matched[
        (crsp_matched.date >= crsp_matched.namedt) & 
        (crsp_matched.date <= crsp_matched.nameendt) &
        crsp_matched.issue_type.isnull() &
        crsp_matched.shrcd.isin([10,11]) &
        crsp_matched.exchcd.isin([1,2,3])]

# Constructing groups of data with different scope to compare
obs_group_1 = int_d_combined[int_d_combined.old_obs == 1]

obs_group_2 = obs_group_1.append(
        int_d_combined[(int_d_combined.old_obs == 0) & (int_d_combined.issue_type == "0")])
obs_group_2 = obs_group_2.append(crsp_matched)

obs_group_3 = obs_group_2.append(
        int_d_combined[(int_d_combined.old_obs == 0) & (int_d_combined.issue_type.isin(["7","F"]))])

obs_group_4 = int_d_combined.append(df_index)

# Saving group 2 and 4
columns_to_save = ["secid","date", "D_clamp", "D_in_sample", "rn_prob_20", "rn_prob_40", "rn_prob_60", "rn_prob_80"]
obs_group_2[columns_to_save].to_csv("estimated_data/interpolated_D/int_ind_disaster_comb_days_" + str(days) + ".csv")
obs_group_4[columns_to_save].to_csv("estimated_data/interpolated_D/int_ind_disaster_comb_all_days_" + str(days) + ".csv")

# obs_group_1 = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_days_" + str(days) + ".csv")
# obs_group_2 = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_comb_days_" + str(days) + ".csv")
# obs_group_4 = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_comb_all_days_" + str(days) + ".csv")

df_issue_type = pd.merge(df_issue_type, df_crsp_cusip.drop(columns = "cusip"), 
        left_on = "cusip", right_on = "ncusip", how = "left")






df_issue_type[df_issue_type.shrcd.isin([10,11]) & 
              df_issue_type.exchcd.isin([1,2,3])][["secid", "issue_type"]].drop_duplicates().groupby("issue_type").secid.count()






# Filtering each of them and constructing monthly averages
def mean_with_truncation(x):
    return np.mean(x[(x <= np.nanquantile(x, 0.975)) & (x >= np.nanquantile(x, 0.025))])

def filter_ind_disaster(df, variable, min_obs_in_month, min_share_month):
    df["date"] = pd.to_datetime(df["date"])
    df["date_eom"] = df["date"] + pd.offsets.MonthEnd(0)
    
    def min_month_obs(x):
        return x[variable].count() >= min_obs_in_month
    
    df_filter_1 = df.groupby(["secid", "date_eom"]).filter(min_month_obs)
    df_mon_mean = df_filter_1.groupby(["secid", "date_eom"]).mean().reset_index()
    
    num_months = len(np.unique(df_mon_mean["date_eom"]))
    def min_sample_obs(x):
        return x[variable].count() > num_months * min_share_month
    
    df_filter = df_mon_mean.groupby("secid").filter(min_sample_obs)
    df_filter = df_filter.rename(columns = {"date_eom": "date"})
    
    return df_filter[["secid", "date", variable]]


#obs_group_1_filter = filter_ind_disaster(obs_group_1, "D_clamp", 1, 0)
#obs_group_2_filter = filter_ind_disaster(obs_group_2, "D_clamp", 1, 0)
#obs_group_3_filter = filter_ind_disaster(obs_group_3, "D_clamp", 1, 0)
#obs_group_4_filter = filter_ind_disaster(obs_group_4, "D_clamp", 1, 0)
#
#D_group_list = [
#        obs_group_1_filter.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_1"),
#        obs_group_2_filter.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_2"),
#        obs_group_3_filter.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_3"),
#        obs_group_4_filter.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_4")]
#
#D_all_groups = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), D_group_list)

obs_group_1_filter_2 = filter_ind_disaster(obs_group_1, "D_clamp", 5, 0)
obs_group_2_filter_2 = filter_ind_disaster(obs_group_2, "D_clamp", 5, 0)
obs_group_3_filter_2 = filter_ind_disaster(obs_group_3, "D_clamp", 5, 0)
obs_group_4_filter_2 = filter_ind_disaster(obs_group_4, "D_clamp", 5, 0)

D_group_list_2 = [
        obs_group_1_filter_2.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_1"),
        obs_group_2_filter_2.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_2"),
        obs_group_3_filter_2.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_3"),
        obs_group_4_filter_2.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_4")]

D_all_groups_2 = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), D_group_list_2)


#obs_group_1_filter_3 = filter_ind_disaster(obs_group_1, "D_clamp", 10, 0)
#obs_group_2_filter_3 = filter_ind_disaster(obs_group_2, "D_clamp", 10, 0)
#obs_group_3_filter_3 = filter_ind_disaster(obs_group_3, "D_clamp", 10, 0)
#obs_group_4_filter_3 = filter_ind_disaster(obs_group_4, "D_clamp", 10, 0)
#
#D_group_list_3 = [
#        obs_group_1_filter_3.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_1"),
#        obs_group_2_filter_3.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_2"),
#        obs_group_3_filter_3.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_3"),
#        obs_group_4_filter_3.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_4")]
#
#D_all_groups_3 = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), D_group_list_3)


########################################################
# Plotting
# Number of firms:
obs_group_1_filter_2.groupby("date").secid.count().plot(label = "Group 1")
obs_group_2_filter_2.groupby("date").secid.count().plot(label = "Group 2")
plt.legend()

obs_group_2_filter_2.groupby("date").secid.count().plot(label = "Group 2")
obs_group_3_filter_2.groupby("date").secid.count().plot(label = "Group 3")
plt.legend()

obs_group_3_filter_2.groupby("date").secid.count().plot(label = "Group 3")
obs_group_4_filter_2.groupby("date").secid.count().plot(label = "Group 4")
plt.legend()

# Panel structure:
(obs_group_1_filter_2.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 1")
(obs_group_2_filter_2.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 2")
plt.legend()

(obs_group_2_filter_2.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 2")
(obs_group_3_filter_2.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 3")
plt.legend()

(obs_group_3_filter_2.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 3")
(obs_group_4_filter_2.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 4")
plt.legend()

# Disaster measures:
D_group_list_2 = [
        obs_group_1_filter_2.groupby("date")["D_clamp"].mean().rename("group_1"),
        obs_group_2_filter_2.groupby("date")["D_clamp"].mean().rename("group_2"),
        obs_group_3_filter_2.groupby("date")["D_clamp"].mean().rename("group_3"),
        obs_group_4_filter_2.groupby("date")["D_clamp"].mean().rename("group_4")]
D_all_groups_2 = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), D_group_list_2)
D_all_groups_2.plot()
D_all_groups_2.diff().corr()

D_group_list_2 = [
        obs_group_1_filter_2.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_1"),
        obs_group_2_filter_2.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_2"),
        obs_group_3_filter_2.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_3"),
        obs_group_4_filter_2.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_4")]
D_all_groups_2 = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), D_group_list_2)
D_all_groups_2.plot()
D_all_groups_2.diff().corr()

D_group_list_2 = [
        obs_group_1_filter_2.groupby("date")["D_clamp"].median().rename("group_1"),
        obs_group_2_filter_2.groupby("date")["D_clamp"].median().rename("group_2"),
        obs_group_3_filter_2.groupby("date")["D_clamp"].median().rename("group_3"),
        obs_group_4_filter_2.groupby("date")["D_clamp"].median().rename("group_4")]
D_all_groups_2 = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), D_group_list_2)
D_all_groups_2.plot()
D_all_groups_2.diff().corr()

# Adding a minimum number of months requirement

obs_group_1_filter_4 = filter_ind_disaster(obs_group_1, "D_clamp", 5, 0.25)
obs_group_2_filter_4 = filter_ind_disaster(obs_group_2, "D_clamp", 5, 0.25)
obs_group_3_filter_4 = filter_ind_disaster(obs_group_3, "D_clamp", 5, 0.25)
obs_group_4_filter_4 = filter_ind_disaster(obs_group_4, "D_clamp", 5, 0.25)

D_group_list_4 = [
        obs_group_1_filter_4.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_1"),
        obs_group_2_filter_4.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_2"),
        obs_group_3_filter_4.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_3"),
        obs_group_4_filter_4.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("group_4")]

D_all_groups_4 = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), D_group_list_4)
D_all_groups_4.plot()

# Comparing different aggregations of disaster measure:

comp_list = [
        obs_group_1_filter_2.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("Trunc Mean"),
        obs_group_1_filter_2.groupby("date")["D_clamp"].median().rename("Median"),
        obs_group_1_filter_4.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("Min Share Trunc Mean")]
comp_df = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), comp_list)

emil_D = pd.read_csv("Emils_D.csv").rename(columns = {"Date":"date"})
emil_D["date"] = pd.to_datetime(emil_D["date"], infer_datetime_format = True) + pd.offsets.MonthEnd(0)
emil_D = emil_D.set_index("date")

comp_df = pd.merge(comp_df, emil_D, left_index = True, right_index = True, how = "left")
comp_df.plot()
comp_df.diff().corr()


comp_list = [
        obs_group_3_filter_2.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("Trunc Mean"),
        obs_group_3_filter_2.groupby("date")["D_clamp"].median().rename("Median"),
        obs_group_3_filter_4.groupby("date")["D_clamp"].apply(mean_with_truncation).rename("Min Share Trunc Mean")]
comp_df = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), comp_list)

emil_D = pd.read_csv("Emils_D.csv").rename(columns = {"Date":"date"})
emil_D["date"] = pd.to_datetime(emil_D["date"], infer_datetime_format = True) + pd.offsets.MonthEnd(0)
emil_D = emil_D.set_index("date")

comp_df = pd.merge(comp_df, emil_D, left_index = True, right_index = True, how = "left")
comp_df.plot()
comp_df.diff().corr()








