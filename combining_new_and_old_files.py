"""
Need to do this another time
"""
import pandas as pd
import numpy as np
from functools import reduce
import wrds
import os
from matplotlib import pyplot as plt
os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision")
import data_funcs
pd.set_option('display.max_columns', None)

for days in [30, 60, 90, 120, 150, 180]:
    print(days)
    file_name = "estimated_data/interpolated_D/int_ind_disaster_days_" + str(days) + ".csv"
    old_int_d = pd.read_csv(file_name)
    old_int_d["date"] = pd.to_datetime(old_int_d["date"])
    old_int_d = old_int_d[["secid", "date", 'D_clamp', 'D_in_sample',"rn_prob_20","rn_prob_40",
                           "rn_prob_60","rn_prob_80"]]
    
    file_name = "estimated_data/interpolated_D/int_ind_disaster_missing_days_" + str(days) + ".csv"
    missing_int_d = pd.read_csv(file_name)
    missing_int_d["date"] = pd.to_datetime(missing_int_d["date"])
    missing_int_d["flag_overlap"] = 1
    
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
    
    # Identifying Common Shares as union of OM and CRSP definitions:
    int_d_combined = data_funcs.get_cs_indicator(int_d_combined)
    int_d_combined = int_d_combined[int_d_combined.cs == 1]
    int_d_combined = int_d_combined.drop(columns = ["cs"])
    int_d_combined.to_csv("estimated_data/interpolated_D/int_ind_disaster_union_cs_" + str(days) + ".csv")

########################################################
# Now looking through a particular measure and compare
# the aggregated disaster measure before and after we
# add new observation
########################################################
#int_d_combined = pd.read_csv("estimated_data/interpolated_D/int_ind_disaster_union_cs_90.csv")
#obs_group_filter = data_funcs.filter_ind_disaster(int_d_combined, "D_clamp", 5, 0)
#obs_group_filter.groupby("date").secid.count().plot()
#D = obs_group_filter.groupby("date")["D_clamp"].apply(data_funcs.mean_with_truncation)
#D.plot()





#D_group_list_90 = [
#        obs_group_1_filter_90.groupby("date")["D_clamp"].apply(data_funcs.mean_with_truncation).rename("group_1"),
#        obs_group_2_filter_90.groupby("date")["D_clamp"].apply(data_funcs.mean_with_truncation).rename("group_2"),
#        obs_group_3_filter_90.groupby("date")["D_clamp"].apply(data_funcs.mean_with_truncation).rename("group_3")]
#
#D_all_groups_90 = reduce(lambda df1, df2: pd.merge(df1, df2, left_index = True, right_index = True), D_group_list_90)
#
#########################################################
## Plotting
#########################################################
#d_to_link.groupby("date").secid.count().plot(label = "all", alpha = 0.7, figsize = (10,6))
#obs_group_1_filter_30.groupby("date").secid.count().plot(label = "Group 1")
#obs_group_2_filter_30.groupby("date").secid.count().plot(label = "Group 2", alpha = 0.7)
#obs_group_3_filter_30.groupby("date").secid.count().plot(label = "Group 3", alpha = 0.7)
#plt.legend()
#
## Panel structure:
#(obs_group_1_filter_30.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 1", figsize = (8,6))
#(obs_group_2_filter_30.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 2")
#(obs_group_3_filter_30.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 3")
#plt.legend()
#
#D_all_groups_30.plot(figsize = (10, 6))
#D_all_groups_30.diff().corr()
#
#
#d_to_link.groupby("date").secid.count().plot(label = "all", alpha = 0.7, figsize = (10,6))
#obs_group_1_filter_90.groupby("date").secid.count().plot(label = "Group 1")
#obs_group_2_filter_90.groupby("date").secid.count().plot(label = "Group 2", alpha = 0.7)
#obs_group_3_filter_90.groupby("date").secid.count().plot(label = "Group 3", alpha = 0.7)
#plt.legend()
#
## Panel structure:
#(obs_group_1_filter_90.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:300].plot(label = "Group 1", figsize = (8,6))
#(obs_group_2_filter_90.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:300].plot(label = "Group 2")
#(obs_group_3_filter_90.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:300].plot(label = "Group 3")
#plt.legend()
#
#D_all_groups_90.plot(figsize = (10, 6))
#D_all_groups_90.diff().corr()
#
#
#
#d_to_link[(d_to_link.crsp_cs == 1) | (d_to_link.issue_type == "0")].groupby("date").secid.count().plot(label = "all", alpha = 0.7, figsize = (10,8), color = "0")
#obs_group_1_filter_30.groupby("date").secid.count().plot(label = "Group 1, 30 day")
#obs_group_2_filter_30.groupby("date").secid.count().plot(label = "Group 2, 30 day", alpha = 0.7)
#obs_group_3_filter_30.groupby("date").secid.count().plot(label = "Group 3, 30 day", alpha = 0.7)
#obs_group_1_filter_90.groupby("date").secid.count().plot(label = "Group 1, 90 day", linestyle = "--")
#obs_group_2_filter_90.groupby("date").secid.count().plot(label = "Group 2, 90 day", alpha = 0.7, linestyle = "--")
#obs_group_3_filter_90.groupby("date").secid.count().plot(label = "Group 3, 90 day", alpha = 0.7, linestyle = "--")
#plt.legend()
#
#(obs_group_1_filter_30.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 1, 30 day", figsize = (8,6))
#(obs_group_2_filter_30.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 2, 30 day")
#(obs_group_3_filter_30.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:200].plot(label = "Group 3, 30 day")
#(obs_group_1_filter_90.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:300].plot(label = "Group 1, 90 day", linestyle = "--")
#(obs_group_2_filter_90.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:300].plot(label = "Group 2, 90 day", linestyle = "--")
#(obs_group_3_filter_90.groupby("secid").date.count().sort_values(ascending = False).reset_index().date/264).iloc[0:300].plot(label = "Group 3, 90 day", linestyle = "--")
#plt.legend()




#
#
# # (1) For each (secid, date) getting the best link from OM-CRSP linking table:
# d_to_link = int_d_combined[["secid", "date"]]
# d_to_link = pd.merge(d_to_link, om_crsp, on = "secid", how = "left")
# not_matched_1 = d_to_link[(d_to_link.score == 6) | d_to_link.score.isnull()]
# d_to_link = d_to_link[(d_to_link.date <= d_to_link.edate) & (d_to_link.date >= d_to_link.sdate)]
# d_to_link = d_to_link.drop(columns = ["sdate", "edate", "score"])
#
# # (2) Now linking with CRSP MSENAMES by PERMNO and filtering by
# # shrcd and exchcd
# d_to_link = pd.merge(d_to_link, df_crsp_names, on = "permno", how = "left")
# d_to_link = d_to_link[(d_to_link.date >= d_to_link.namedt) & (d_to_link.date <= d_to_link.nameendt)]
# d_to_link["crsp_cs"] = np.where(d_to_link.shrcd.isin([10,11]) & d_to_link.exchcd.isin([1,2,3]), 1, 0)
# d_to_link = d_to_link.drop(columns = ["namedt", "nameendt", "shrcd", "exchcd"])
#
# # (3) Merging with OM issue_types
# d_to_link = pd.merge(d_to_link, df_issue_type, on = "secid", how = "left")
#
# obs_group_1 = d_to_link[d_to_link.issue_type == "0"]
# obs_group_2 = d_to_link[d_to_link.crsp_cs == 1]
# obs_group_3 = d_to_link[(d_to_link.issue_type == "0") | (d_to_link.crsp_cs == 1)]
#
# # Merging disaster measure:
# obs_group_1 = pd.merge(obs_group_1, int_d_combined, on = ["secid", "date"])
# obs_group_2 = pd.merge(obs_group_2, int_d_combined, on = ["secid", "date"])
# obs_group_3 = pd.merge(obs_group_3, int_d_combined, on = ["secid", "date"])
